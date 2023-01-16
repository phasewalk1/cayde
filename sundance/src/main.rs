use argh::FromArgs;
use serde_json::json;
use std::env;
use walkdir::WalkDir;

mod encoders;
use encoders::SpectroOneHotEncoder as SundanceEncoder;

#[derive(FromArgs)]
#[argh(description = "Sundance Image Encoder")]
struct SundanceCli {
    #[argh(option, short = 'd')]
    /// directory to scan for images
    img_dir: String,
}

fn main() {
    let args: SundanceCli = argh::from_env();
    let img_dir = args.img_dir;

    let mut out = vec![vec![]];
    // walk the directory and find all images
    for img_path in WalkDir::new(img_dir) {
        let img_path = img_path.unwrap().path().to_str().unwrap().to_string();
        if img_path.ends_with(".jpg") || img_path.ends_with(".png") {
            println!("Found image: {}", img_path);
            let mut encoder = SundanceEncoder::new(&img_path);
            let encoded = encoder.encode();
            out.push(encoded);
        }
    }

    println!("Jsonifying output ...");
    let out_json = serde_json::to_string_pretty(&json!(out))
        .map_err(|e| panic!("Failed to serialize output to json: {}", e))
        .unwrap();
    let out_path = env::current_dir().unwrap().join("encodings.json");
    println!("Writing encodings to disk ... : {:?}", out_path);
    std::fs::write(out_path, out_json).unwrap();
}
