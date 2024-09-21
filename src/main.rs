use core::f64;
use std::{collections::HashMap, sync::{Arc, Mutex}};
use image::{ImageBuffer, Pixel, Rgba};
use rayon::prelude::*;

mod GP;

fn main() {
    image_evolution();
}

#[derive(Debug, Clone)]
struct PaintGene<'a> {
    top: i64,
    left: i64,
    scale: u8,
    opacity: u8,
    pattern: u8,
    max_scale: u8,
    max_opacity: u8,
    max_pattern: u8,
    lut: &'a HashMap<(u8, u8, u8), ImageBuffer<Rgba<u8>, Vec<u8>>>,
}

impl PaintGene<'_> {
    fn mutate(&mut self, canvas_width: u32, canvas_height: u32) {
        let mutation_degree = fastrand::u8(0..11);
        
        // Define the boundary limits (1/4th of the canvas size)
        let boundary_width = canvas_width as i64 / 4;
        let boundary_height = canvas_height as i64 / 4;
        
        // Define the max allowed ranges based on the boundaries
        let min_top = -boundary_height;
        let max_top = (canvas_height as i64) + boundary_height;
        let min_left = -boundary_width;
        let max_left = (canvas_width as i64) + boundary_width;

        if mutation_degree > 2 {
            // Adjust top within boundaries
            self.top = (self.top + fastrand::i64(0..201) - 100).clamp(min_top, max_top);
            // Adjust left within boundaries
            self.left = (self.left + fastrand::i64(0..201) - 100).clamp(min_left, max_left);
        }
        if mutation_degree > 4 {
            if fastrand::bool() && self.opacity < self.max_opacity {
                self.opacity += 1;
            } else if self.opacity > 0 {
                self.opacity -= 1;
            }
        }
        if mutation_degree > 7 {
            if fastrand::bool() && self.scale < self.max_scale {
                self.scale += 1;
            } else if self.scale > 0 {
                self.scale -= 1;
            }
        }
        if mutation_degree == 10 {
            if fastrand::bool() && self.pattern < self.max_pattern {
                self.pattern += 1;
            } else if self.pattern > 0 {
                self.pattern -= 1;
            }
        }
    }

    fn paint(&self, canvas: &mut ImageBuffer<Rgba<u8>, Vec<u8>>) {
        if let Some(pattern) = self.lut.get(&(self.pattern, self.scale, self.opacity)) {
            image::imageops::overlay(canvas, pattern, self.top, self.left);
        }
    }
}

#[derive(Debug)]
struct PaintGenome<'a> {
    score: u128,
    genes: Vec<PaintGene<'a>>,
    lut: &'a HashMap<(u8, u8, u8), ImageBuffer<Rgba<u8>, Vec<u8>>>,
    max_scale: u8,
    max_opacity: u8,
    max_pattern: u8,
    max_width: u32,
    max_height: u32,
    cache_paint: ImageBuffer<Rgba<u8>, Vec<u8>>,
    valid_cache: bool
}

impl<'a> PaintGenome<'a> {
    fn mutate(&mut self) {
        self.genes.iter_mut().for_each(|gene| gene.mutate(self.max_width, self.max_height));
        self.valid_cache = false;
    }

    fn crossover(&mut self, other: &mut Self) {
        let gene_count = self.genes.len();
        let transfer_count = 5;
        let mut rng = fastrand::Rng::new();
        let transfer_to = rng.choose_multiple(0..gene_count, transfer_count);
        let transfer_from = rng.choose_multiple(0..other.genes.len(), transfer_count);

        // Use Vec for indexing
        let self_genes: Vec<_> = transfer_to.iter().map(|&i| self.genes[i].clone()).collect();
        let other_genes: Vec<_> = transfer_from.iter().map(|&i| other.genes[i].clone()).collect();

        self.genes.extend(self_genes.into_iter());
        other.genes.extend(other_genes.into_iter());

        // Removing genes
        for from in transfer_to {
            self.genes.remove(from);
        }
        for to in transfer_from {
            other.genes.remove(to);
        }

        self.valid_cache = false;
    }

    fn gen_children(&self, child_count: u8) -> Vec<Self> {
        (0..child_count)
            .map(|_| PaintGenome {
                score: self.score,
                genes: self.genes.iter().cloned().collect(),
                lut: self.lut,
                max_scale: self.max_scale,
                max_opacity: self.max_opacity,
                max_pattern: self.max_pattern,
                max_height: self.max_height,
                max_width: self.max_width,
                cache_paint: ImageBuffer::new(0,0),
                valid_cache: false
            })
            .collect()
    }

    fn new(lut: &'a HashMap<(u8, u8, u8), ImageBuffer<Rgba<u8>, Vec<u8>>>, gene_count: u32, max_opacity: u8, max_scale: u8, max_pattern: u8, max_width: u32, max_height: u32) -> Self {
        let genes = (0..gene_count).map(|_| {
            PaintGene {
                top: fastrand::i64(0..max_height as i64),
                left: fastrand::i64(0..max_width as i64),
                scale: fastrand::u8(0..max_scale),
                opacity: fastrand::u8(0..max_opacity),
                pattern: fastrand::u8(0..max_pattern),
                lut,
                max_opacity,
                max_pattern,
                max_scale,
            }
        }).collect();

        PaintGenome {
            score: u128::MAX,
            genes,
            lut,
            max_scale,
            max_opacity,
            max_pattern,
            max_width,
            max_height,
            cache_paint: ImageBuffer::new(0,0),
            valid_cache: false
        }
    }

    fn paint(&mut self) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        if self.valid_cache {
            return self.cache_paint.clone();
        }

        let mut canvas = ImageBuffer::from_fn(self.max_width, self.max_height, |_, _| Rgba([255, 255, 255, 255]));
        for gene in &self.genes {
            gene.paint(&mut canvas);
        }
        self.valid_cache = true;
        self.cache_paint = canvas.clone(); // TODO this is stupid, just make the eval use the last score, the target image doesn't change
        canvas
    }
}

// Calculation functions
pub fn pixel_absd(image_a: &Rgba<u8>, image_b: &Rgba<u8>) -> i32 {
    image_a.channels().iter()
        .zip(image_b.channels().iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .sum()
}

pub fn similarity_sad(image_a: &ImageBuffer<Rgba<u8>, Vec<u8>>, image_b: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> u128 {
    (0..image_a.width())
        .flat_map(|x| (0..image_a.height()).map(move |y| (x, y)))
        .map(|(x, y)| pixel_absd(image_a.get_pixel(x, y), image_b.get_pixel(x, y)) as u128)
        .sum()
}

fn image_eval(genome: &mut PaintGenome, target_image: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> u128 {
    let canvas = genome.paint();
    similarity_sad(&canvas, target_image)
}

fn adjust_opacity(image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, opacity: u8, max_opacity: u8) {
    let alpha_factor = opacity as f64 / max_opacity as f64;
    
    image.pixels_mut().for_each(|pixel| {
        let Rgba([r, g, b, a]) = *pixel;
        let new_alpha = (a as f64 * alpha_factor).round() as u8;
        *pixel = Rgba([r, g, b, new_alpha]);
    });
}

/*
fn greedy_image_search() {
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

    let canvas = ImageBuffer::from()
    
    for iteration in 0..iterations {
        // Evaluate
        
        // Selection
        population.sort_by(|a, b| a.score.cmp(&b.score));
        population.truncate(5);
        println!("Iteration: {}, Best Score: {}", iteration, population[0].score);
        
        // Save image
        let best = population[0].paint();
        best.save(format!("{}/{}.png", output_folder, iteration)).expect("Failed to save image");

        
        // Population growth
        let mut new_population_members: Vec<PaintGenome> = Vec::new();
        population.iter().for_each(|x| {
            new_population_members.extend(x.gen_children(children_per_iteration));
        });
        
        // Crossover
        let cuttoff = new_population_members.len()/2;
        let (pop_slice_1, pop_slice_2) = new_population_members.split_at_mut(cuttoff);
        let lucky_one = &mut pop_slice_1[fastrand::usize(..pop_slice_1.len())];
        let lucky_two = &mut pop_slice_2[fastrand::usize(..pop_slice_2.len())];
        lucky_one.crossover(lucky_two);

        new_population_members.par_iter_mut().for_each(|pop| pop.mutate());

        population.extend(new_population_members);
    }
    
    let best_score = population.iter().map(|x| x.score).min().unwrap_or(u128::MAX);
    println!("Final Best Score: {}", best_score);
}
*/

// toooo slooooooooow
fn image_evolution() {
    let image_path = "4.jpg";
    let pattern_paths = ["p1.png", "p2.png", "p3.png"];

    let image = image::open(image_path)
        .expect("Target image not found at provided path")
        .to_rgba8();

    let patterns: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> = pattern_paths.iter()
        .map(|&path| image::open(path).expect("Pattern does not exist").to_rgba8())
        .collect();

    let size_levels = 5;
    let opacity_levels = 10;
    let pattern_count = patterns.len() as u8;
    let gene_count = 200;
    let population_size = 5;
    let children_per_iteration = 2;
    let iterations = 1000;

    let output_folder = "./images";

    // Create LUT
    let mut lut = HashMap::new();
    for (index, pattern) in patterns.iter().enumerate() {
        for size in 0..size_levels {
            for opac in 0..opacity_levels {
                let mut scale_modifier = 1.0; // TODO all too
                let modified_scale = (size+1)-5;
                if modified_scale < 0 {
                    scale_modifier = 1.0/(scale_modifier as f64).abs()
                }
                else {
                    scale_modifier = (size+1) as f64; // Improvement: Scale up or down
                }
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

    let mut population: Vec<PaintGenome> = (0..population_size)
        .map(|_| PaintGenome::new(&lut, gene_count, opacity_levels - 1, size_levels - 1, pattern_count - 1, image.width(), image.height()))
        .collect();

    for iteration in 0..iterations {
        // Evaluate
        population.par_iter_mut().for_each(|pop| {
            pop.score = image_eval(pop, &image);
        });

        // Selection
        population.sort_by(|a, b| a.score.cmp(&b.score));
        population.truncate(5);
        println!("Iteration: {}, Best Score: {}", iteration, population[0].score);

        // Save image
        let best = population[0].paint();
        best.save(format!("{}/{}.png", output_folder, iteration)).expect("Failed to save image");

        
        // Population growth
        let mut new_population_members: Vec<PaintGenome> = Vec::new();
        population.iter().for_each(|x| {
            new_population_members.extend(x.gen_children(children_per_iteration));
        });
        
        // Crossover
        let cuttoff = new_population_members.len()/2;
        let (pop_slice_1, pop_slice_2) = new_population_members.split_at_mut(cuttoff);
        let lucky_one = &mut pop_slice_1[fastrand::usize(..pop_slice_1.len())];
        let lucky_two = &mut pop_slice_2[fastrand::usize(..pop_slice_2.len())];
        lucky_one.crossover(lucky_two);

        new_population_members.par_iter_mut().for_each(|pop| pop.mutate());

        population.extend(new_population_members);
    }

    let best_score = population.iter().map(|x| x.score).min().unwrap_or(u128::MAX);
    println!("Final Best Score: {}", best_score);
}
