use image::{DynamicImage, GenericImageView};
use ndarray::Array2;

#[derive(Default)]
pub struct Encoder {
    pixels: Array2<u8>,
}

impl Encoder {
    pub fn new(image_path: &str) -> Self {
        println!("Preparing encoder for {}", image_path);
        let img: DynamicImage = image::open(image_path).unwrap();

        let (width, height) = img.dimensions();
        println!("Img Dim --> {}x{}", width, height);

        let mut pixels = Array2::default((width as usize, height as usize));
        println!("Mat Shape --> {:?}", pixels.shape());

        for (x, y, pixel) in img.pixels() {
            pixels[[x as usize, y as usize]] = pixel.0;
        }

        // extract the u8 values from the pixels
        let pixels = pixels.mapv(|value| value[0]);

        return Encoder { pixels };
    }

    pub fn encode(&self) -> Vec<usize> {
        let mut encoded = Vec::new();
        for row in self.pixels.rows() {
            let mut encoded_row = Vec::new();
            for pixel in row {
                let encoded_pixel: usize = match pixel {
                    1 => 1,
                    _ => 0,
                };
                encoded_row.push(encoded_pixel);
            }
            encoded.extend(encoded_row);
        }
        return encoded;
    }
}
