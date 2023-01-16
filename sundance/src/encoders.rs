use image::{DynamicImage, GenericImageView};
use ndarray::Array2;

#[derive(Default)]
pub struct SpectroOneHotEncoder {
    pixels: Array2<u8>,
}

impl SpectroOneHotEncoder {
    pub fn new(image_path: &str) -> Self {
        pretty_env_logger::try_init().ok();

        log::info!("Preparing encoder for {}", image_path);
        let img: DynamicImage = image::open(image_path).unwrap();

        let (width, height) = img.dimensions();
        log::debug!("Img Dim --> {}x{}", width, height);

        let mut pixels = Array2::default((width as usize, height as usize));
        log::debug!("Mat Shape --> {:?}", pixels.shape());

        for (x, y, pixel) in img.pixels() {
            pixels[[x as usize, y as usize]] = pixel.0;
        }

        // extract the u8 values from the pixels
        let pixels = pixels.mapv(|value| value[0]);

        return SpectroOneHotEncoder { pixels };
    }

    pub fn encode(&mut self) -> Vec<[usize; 2]> {
        log::info!("Encoding ...");

        let mut encoded = Vec::new();
        for row in self.pixels.rows() {
            let mut encoded_row = Vec::new();
            for pixel in row {
                let encoded_pixel: [usize; 2] = match pixel {
                    1 => [0 as usize, 1 as usize],
                    _ => [1 as usize, 0 as usize],
                };
                encoded_row.push(encoded_pixel);
            }
            encoded.extend(encoded_row);
        }

        return encoded;
    }
}
