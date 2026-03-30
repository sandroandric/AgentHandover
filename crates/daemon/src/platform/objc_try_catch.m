// objc_try_catch.m — Objective-C helpers for Rust FFI.
//
// All Vision framework calls live here so that ObjC exceptions never
// propagate through Rust stack frames (which would abort the process).

#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Vision/Vision.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ──────────────────────────────────────────────────────────
// C-compatible result structs returned to Rust
// ──────────────────────────────────────────────────────────

/// A single recognized text element.
typedef struct {
    char *text;           // heap-allocated UTF-8, caller frees
    float confidence;
    double bbox_x;        // normalized 0.0-1.0 (top-left origin)
    double bbox_y;
    double bbox_w;
    double bbox_h;
} OcrElement;

/// Full OCR result from a single image.
typedef struct {
    OcrElement *elements; // heap-allocated array, caller frees each .text then the array
    int count;
    char *full_text;      // heap-allocated UTF-8, caller frees
    int success;          // 1 = success, 0 = failure
} OcrCResult;

// ──────────────────────────────────────────────────────────
// Vision OCR — entire pipeline in ObjC, exception-safe
// ──────────────────────────────────────────────────────────

/// Perform OCR on raw BGRA pixel data using Apple Vision framework.
/// All ObjC work happens here so exceptions are caught before reaching Rust.
OcrCResult perform_ocr_safe(
    const uint8_t *pixels,
    size_t width,
    size_t height
) {
    OcrCResult result = { .elements = NULL, .count = 0, .full_text = NULL, .success = 0 };

    @autoreleasepool {
        @try {
            // 1. Create CGImage from raw BGRA pixels
            CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
            if (!colorSpace) return result;

            size_t bytesPerRow = width * 4;
            // kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little = 0x2002
            uint32_t bitmapInfo = 0x2002;

            CGContextRef ctx = CGBitmapContextCreate(
                (void *)pixels, width, height, 8, bytesPerRow, colorSpace, bitmapInfo
            );
            CGColorSpaceRelease(colorSpace);
            if (!ctx) return result;

            CGImageRef cgImage = CGBitmapContextCreateImage(ctx);
            CGContextRelease(ctx);
            if (!cgImage) return result;

            // 2. Create VNImageRequestHandler (this is where exceptions may be thrown)
            VNImageRequestHandler *handler = [[VNImageRequestHandler alloc]
                initWithCGImage:cgImage
                    orientation:kCGImagePropertyOrientationUp
                        options:@{}];
            CGImageRelease(cgImage);

            if (!handler) return result;

            // 3. Create and configure VNRecognizeTextRequest
            VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
            request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;

            // 4. Perform the request
            NSError *error = nil;
            BOOL ok = [handler performRequests:@[request] error:&error];
            if (!ok || error) {
                return result;
            }

            // 5. Extract results
            NSArray<VNRecognizedTextObservation *> *observations = request.results;
            NSUInteger obsCount = observations.count;

            if (obsCount == 0) {
                result.success = 1;
                result.full_text = strdup("");
                return result;
            }

            result.elements = (OcrElement *)calloc(obsCount, sizeof(OcrElement));
            if (!result.elements) return result;

            NSMutableArray<NSString *> *textParts = [NSMutableArray arrayWithCapacity:obsCount];
            int validCount = 0;

            for (NSUInteger i = 0; i < obsCount; i++) {
                VNRecognizedTextObservation *obs = observations[i];
                NSArray<VNRecognizedText *> *candidates = [obs topCandidates:1];
                if (candidates.count == 0) continue;

                VNRecognizedText *top = candidates[0];
                NSString *text = top.string;
                if (!text || text.length == 0) continue;

                const char *utf8 = [text UTF8String];
                if (!utf8) continue;

                // Bounding box: Vision uses bottom-left origin, convert to top-left
                CGRect bbox = obs.boundingBox;
                OcrElement *elem = &result.elements[validCount];
                elem->text = strdup(utf8);
                elem->confidence = top.confidence;
                elem->bbox_x = bbox.origin.x;
                elem->bbox_y = 1.0 - bbox.origin.y - bbox.size.height;
                elem->bbox_w = bbox.size.width;
                elem->bbox_h = bbox.size.height;

                [textParts addObject:text];
                validCount++;
            }

            result.count = validCount;

            // Build full text
            NSString *fullText = [textParts componentsJoinedByString:@"\n"];
            result.full_text = strdup([fullText UTF8String] ?: "");
            result.success = 1;
        }
        @catch (NSException *exception) {
            fprintf(stderr, "[AgentHandover] Caught ObjC exception in OCR: %s — %s\n",
                    [[exception name] UTF8String],
                    [[exception reason] UTF8String]);
            // Clean up partial results
            if (result.elements) {
                for (int i = 0; i < result.count; i++) {
                    free(result.elements[i].text);
                }
                free(result.elements);
                result.elements = NULL;
            }
            if (result.full_text) {
                free(result.full_text);
                result.full_text = NULL;
            }
            result.count = 0;
            result.success = 0;
        }
        @catch (...) {
            fprintf(stderr, "[AgentHandover] Caught unknown exception in OCR\n");
            if (result.elements) {
                for (int i = 0; i < result.count; i++) {
                    free(result.elements[i].text);
                }
                free(result.elements);
                result.elements = NULL;
            }
            if (result.full_text) {
                free(result.full_text);
                result.full_text = NULL;
            }
            result.count = 0;
            result.success = 0;
        }
    }

    return result;
}

// ──────────────────────────────────────────────────────────
// NSPasteboard helpers — exception-safe wrappers
// ──────────────────────────────────────────────────────────

/// Pasteboard metadata returned to Rust.
typedef struct {
    long long change_count;
    char **types;       // NULL-terminated array of heap-allocated UTF-8 strings
    int type_count;
    uint8_t *data;      // heap-allocated data for first supported type
    size_t data_len;
    int success;        // 1 = success, 0 = failure
} PasteboardInfo;

/// Get pasteboard change count. Returns -1 on failure.
long long pasteboard_change_count_safe(void) {
    @autoreleasepool {
        @try {
            NSPasteboard *pb = [NSPasteboard generalPasteboard];
            if (!pb) return -1;
            return (long long)[pb changeCount];
        }
        @catch (NSException *exception) {
            fprintf(stderr, "[AgentHandover] Caught ObjC exception in pasteboard changeCount: %s\n",
                    [[exception reason] UTF8String]);
            return -1;
        }
        @catch (...) {
            fprintf(stderr, "[AgentHandover] Caught unknown exception in pasteboard changeCount\n");
            return -1;
        }
    }
}

/// Get pasteboard types. Caller must free each string and the array.
PasteboardInfo pasteboard_get_info_safe(void) {
    PasteboardInfo info = { .change_count = -1, .types = NULL, .type_count = 0,
                            .data = NULL, .data_len = 0, .success = 0 };

    @autoreleasepool {
        @try {
            NSPasteboard *pb = [NSPasteboard generalPasteboard];
            if (!pb) return info;

            info.change_count = (long long)[pb changeCount];

            // Get types
            NSArray<NSPasteboardType> *types = [pb types];
            if (!types) {
                info.success = 1;
                return info;
            }

            NSUInteger count = types.count;
            info.types = (char **)calloc(count + 1, sizeof(char *)); // NULL-terminated
            if (!info.types) return info;

            int validCount = 0;
            for (NSUInteger i = 0; i < count; i++) {
                const char *utf8 = [types[i] UTF8String];
                if (utf8) {
                    info.types[validCount++] = strdup(utf8);
                }
            }
            info.type_count = validCount;

            // Get data for first type that has content
            for (NSUInteger i = 0; i < count; i++) {
                NSData *data = [pb dataForType:types[i]];
                if (data && data.length > 0) {
                    info.data_len = (size_t)data.length;
                    info.data = (uint8_t *)malloc(info.data_len);
                    if (info.data) {
                        memcpy(info.data, data.bytes, info.data_len);
                    }
                    break;
                }
            }

            info.success = 1;
        }
        @catch (NSException *exception) {
            fprintf(stderr, "[AgentHandover] Caught ObjC exception in pasteboard info: %s\n",
                    [[exception reason] UTF8String]);
            info.success = 0;
        }
        @catch (...) {
            fprintf(stderr, "[AgentHandover] Caught unknown exception in pasteboard info\n");
            info.success = 0;
        }
    }
    return info;
}

/// Free a PasteboardInfo.
void free_pasteboard_info(PasteboardInfo *info) {
    if (!info) return;
    if (info->types) {
        for (int i = 0; i < info->type_count; i++) {
            free(info->types[i]);
        }
        free(info->types);
        info->types = NULL;
    }
    if (info->data) {
        free(info->data);
        info->data = NULL;
    }
    info->type_count = 0;
    info->data_len = 0;
}

// ──────────────────────────────────────────────────────────
// Cleanup helpers
// ──────────────────────────────────────────────────────────

/// Free an OcrCResult returned by perform_ocr_safe.
void free_ocr_result(OcrCResult *r) {
    if (!r) return;
    if (r->elements) {
        for (int i = 0; i < r->count; i++) {
            free(r->elements[i].text);
        }
        free(r->elements);
        r->elements = NULL;
    }
    if (r->full_text) {
        free(r->full_text);
        r->full_text = NULL;
    }
    r->count = 0;
    r->success = 0;
}
