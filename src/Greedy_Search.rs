

pub fn greedy_image_search() {
    let image_path = "4.jpg";
    let pattern_paths = ["p1.png", "p2.png", "p3.png"];
    
    let image = image::open(image_path)
        .expect("Target image not found at provided path")
        .to_rgba8();

    let patterns: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> = pattern_paths.iter()
        .map(|&path| image::open(path).expect("Pattern does not exist").to_rgba8())
        .collect();

    let size_levels = 5;
    let opacity_levels = 5;
    let pattern_count = patterns.len() as u8;
    let iterations = 200;
    let considerations_per_iteration = 20;

    let output_folder = "./images";

    // Create LUT
    let mut lut = HashMap::new();
    for (index, pattern) in patterns.iter().enumerate() {
        for size in 0..size_levels {
            for opac in 0..opacity_levels {
                let scale_modifier: f64 = (size+1) as f64; // Improvement: Scale up or down
                let new_width = (pattern.width() as f64 * scale_modifier) as u32;
                let new_height = (pattern.height() as f64 * scale_modifier) as u32;
                let mut modified_image = pattern.clone();
                //modified_image.pixels_mut().for_each(|x| x.apply_with_alpha(|x| x, |y| (y as f64 * (opac as f64 * 255.0 / opacity_levels as f64)) as u8));
                //modified_image = image::imageops::resize(&modified_image, new_width, new_height, image::imageops::FilterType::Nearest);

                // Resize the image first
                modified_image = image::imageops::resize(&modified_image, new_width, new_height, image::imageops::FilterType::Nearest);

                // Adjust opacity
                adjust_opacity(&mut modified_image, opac, opacity_levels);
                
                modified_image.save(format!("{}/pattern_debug_{}.png", output_folder, index.to_string()+&size.to_string()+&opac.to_string()));
                lut.insert((index as u8, size, opac), modified_image);
            }
        }
    }
    
    println!("LUT created");

    for iteration in 0..iterations {
        
    }
}