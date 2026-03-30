fn main() {
    // Compile the Objective-C Vision OCR helper.
    // All Vision framework calls live in ObjC so that ObjC exceptions
    // never propagate through Rust stack frames (which would abort).
    cc::Build::new()
        .file("src/platform/objc_try_catch.m")
        .flag("-fobjc-arc")
        .flag("-fobjc-exceptions")
        .compile("objc_try_catch");

    // Link frameworks needed by objc_try_catch.m
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Vision");
    println!("cargo:rustc-link-lib=framework=AppKit"); // NSPasteboard
}
