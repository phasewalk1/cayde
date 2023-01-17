#[rustfmt::skip]
#[allow(unused_imports)] use argh::{FromArgValue, FromArgs};
use serde_json::json;
use std::{env, str::FromStr};
use walkdir::WalkDir;

mod encoders;
use encoders::SpectroOneHotEncoder as SundanceEncoder;
mod builders;
use builders::MelSpectroBuilder as SundanceBuilder;

#[doc = "Sundance mode of operation"]
#[derive(Debug)]
#[allow(non_camel_case_types)]
enum SundanceMode {
    ENCODE,
    SPECTRO,
}

impl FromStr for SundanceMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "encode" => Ok(SundanceMode::ENCODE),
            "spectro" => Ok(SundanceMode::SPECTRO),
            _ => Err(format!("{} is not a valid mode", s)),
        }
    }
}

#[derive(FromArgs)]
#[argh(description = "Sundance builders and encoders")]
struct SundanceCli {
    #[argh(positional)]
    /// mode of operation
    mode: SundanceMode,

    #[argh(option, short = 'd')]
    /// directory to scan for images
    img_dir: String,
}

impl SundanceCli {
    fn encode(&self) {
        log::info!("Running in encode mode ...");

        let img_dir = self.img_dir.clone();
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

    #[allow(unused_variables)]
    fn spectro(&self) {
        log::info!("Running in spectro mode ...");

        const N_FFT: usize = 2048;
        const HOP_LENGTH: usize = 512;
        const N_MELS: usize = 128;
        const SAMPLE_RATE: usize = 44100;
        const N_SAMPLES: usize = 44100;

        let (signal, sr, ns) =
            SundanceBuilder::openwav("example-train/GTZAN-Reduced/blues/blues.00000.wav");

        #[rustfmt::skip]
        let builder = SundanceBuilder::new(
            N_FFT, HOP_LENGTH, N_MELS, SAMPLE_RATE, N_SAMPLES, None, None
        );

        todo!();
    }

    fn run(&self) {
        pretty_env_logger::try_init().ok();

        match self.mode {
            SundanceMode::ENCODE => self.encode(),
            SundanceMode::SPECTRO => self.spectro(),
        }
    }
}

fn main() {
    let sundance: SundanceCli = argh::from_env();
    sundance.run();
}
