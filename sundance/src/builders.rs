#![allow(dead_code)]

use dsp::window::Window;
use hound::WavReader;
use rustfft::{num_complex::Complex, num_traits::Zero, FftPlanner};

#[doc = "A struct that builds a spectrogram from a wav file"]
#[derive(Debug)]
pub struct MelSpectroBuilder {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    sample_rate: usize,
    n_samples: usize,
    window: Option<Window>,
    samples: Option<Vec<f32>>,
}

impl MelSpectroBuilder {
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        sample_rate: usize,
        n_samples: usize,
        window: Option<Window>,
        samples: Option<Vec<f32>>,
    ) -> MelSpectroBuilder {
        return MelSpectroBuilder {
            n_fft,
            hop_length,
            n_mels,
            sample_rate,
            n_samples,
            window,
            samples,
        };
    }

    pub fn push_samples(&mut self, samples: Vec<f32>) {
        self.samples = Some(samples);
    }

    pub fn openwav(path: &str) -> (Vec<f32>, usize, usize) {
        let mut reader = WavReader::open(path).unwrap();
        let samples = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / std::i16::MAX as f32)
            .collect::<Vec<_>>();
        let sample_rate = reader.spec().sample_rate as usize;
        let n_samples = samples.len();

        return (samples, sample_rate, n_samples);
    }

    pub fn forwardfft(&mut self, samples: Option<Vec<f32>>) -> Vec<Complex<f32>> {
        if self.samples == None {
            if samples == None {
                panic!("No samples provided");
            } else {
                self.samples = samples;
            }
        }

        let samp = self.samples.as_ref().unwrap();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(samp.len());
        let mut buffer = vec![Complex::zero(); samp.len()];
        for (i, &sample) in samp.iter().enumerate() {
            buffer[i] = Complex::new(sample, 0.0);
        }
        fft.process(&mut buffer);
        return buffer;
    }

    pub fn hannfunc(samples: Vec<f32>) -> Vec<f32> {
        let window = Window { samples };
        let mut windowed_samples = vec![];
        window.apply(&window.samples, &mut windowed_samples);

        return windowed_samples;
    }
}
