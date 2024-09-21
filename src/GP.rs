pub trait Gene {
    fn mutate(&mut self);
}

pub trait Genome {
    fn mutate(&mut self);
    fn crossover(&mut self, other: &mut Self);
    fn gen_children(&self, child_count: u8) -> Vec<Box<Self>>;
}
