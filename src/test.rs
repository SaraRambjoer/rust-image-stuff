
fn evolution_test() {
    // constants
    let target_sum = 2473;
    let population_size = 5;
    let children_per_iteration = 3;
    let iterations = 1000;
    
    let mut population: Vec<Numbers> = Vec::new();
    
    // init
    for _ in 0..population_size {
        population.push(Numbers::new());
    }

    // loop
    for iteration in 0..iterations {
        // eval
        for pop in &mut population {
            pop.score = eval(&pop, target_sum);
        }
        
        // selection
        let mut scores: Vec<f64> = population.iter().map(|x| x.score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let nth_highest_score = scores[population_size-1];
        population = population.iter().filter(|x| x.score <= nth_highest_score).map(|x| *x).collect();
        population = population[0..5].to_vec();
        //println!("Iteration: {}, Best Score: {}", iteration, scores[0]);
        // crossover
        let mut lucky_one = population[fastrand::usize(..population_size)];
        let mut lucky_two = population[fastrand::usize(..population_size)];
        lucky_one.crossover(&mut lucky_two);

        // population growth
        let mut new_population_members: Vec<Numbers> = Vec::new();
        for pop in &population {
            let children = pop.gen_children(children_per_iteration);
            for child in children {
                new_population_members.push(*child);
            }
        }
        for pop in new_population_members {
            population.push(pop);
        }

        // mutation
        for pop in &mut population {
            pop.mutate();
        }
    }
    // eval
    for pop in &mut population {
        pop.score = eval(&pop, target_sum);
    }
    
    // selection
    let mut scores: Vec<f64> = population.iter().map(|x| x.score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let nth_highest_score = scores[population_size-1];
    println!("Iteration: {}, Best Score: {}", "Last!", scores[0]);
}
fn evolution_test() {
    // constants
    let target_sum = 2473;
    let population_size = 5;
    let children_per_iteration = 3;
    let iterations = 1000;
    
    let mut population: Vec<Numbers> = Vec::new();
    
    // init
    for _ in 0..population_size {
        population.push(Numbers::new());
    }

    // loop
    for iteration in 0..iterations {
        // eval
        for pop in &mut population {
            pop.score = eval(&pop, target_sum);
        }
        
        // selection
        let mut scores: Vec<f64> = population.iter().map(|x| x.score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let nth_highest_score = scores[population_size-1];
        population = population.iter().filter(|x| x.score <= nth_highest_score).map(|x| *x).collect();
        population = population[0..5].to_vec();
        //println!("Iteration: {}, Best Score: {}", iteration, scores[0]);
        // crossover
        let mut lucky_one = population[fastrand::usize(..population_size)];
        let mut lucky_two = population[fastrand::usize(..population_size)];
        lucky_one.crossover(&mut lucky_two);

        // population growth
        let mut new_population_members: Vec<Numbers> = Vec::new();
        for pop in &population {
            let children = pop.gen_children(children_per_iteration);
            for child in children {
                new_population_members.push(*child);
            }
        }
        for pop in new_population_members {
            population.push(pop);
        }

        // mutation
        for pop in &mut population {
            pop.mutate();
        }
    }
    // eval
    for pop in &mut population {
        pop.score = eval(&pop, target_sum);
    }
    
    // selection
    let mut scores: Vec<f64> = population.iter().map(|x| x.score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let nth_highest_score = scores[population_size-1];
    println!("Iteration: {}, Best Score: {}", "Last!", scores[0]);
}

#[derive(Clone, Copy)]
struct Number {
    value: i64,
}

impl Gene for Number {
    fn mutate(&mut self) {
        if fastrand::bool() {
            self.value += 1;
        }
        else {
            self.value -= 1;
        }
    }
}

impl Number {
    fn new() -> Self {
        return Number { value: fastrand::i64(..200) }
    }
}

#[derive(Clone, Copy)]
struct Numbers {
    value: [Number; 10],
    score: f64
}

impl Genome for Numbers {
    fn mutate(&mut self) {
        for val in &mut self.value {
            val.mutate();
        }
    }

    fn crossover(&mut self, other: &mut Self) {
        for num in 0..3 {
            let index1 = fastrand::usize(..10);
            let index2 = fastrand::usize(..10);
            if fastrand::bool() {
                self.value[index1].value = other.value[index2].value;
            }
            else {
                other.value[index1].value = self.value[index2].value;
            }
        }
    }

    fn gen_children(&self, child_count: u8) -> Vec<Box<Numbers>> {
        let mut to_return: Vec<Box<Numbers>> = Vec::new();
        for _ in 0..child_count {
            to_return.push(Box::new(self.clone()));
        }
        
        to_return
    }
}

impl Numbers {
    fn new() -> Self {
        let numbers: [Number; 10] = [Number { value: fastrand::i64(0..100)}; 10];

        Numbers { value: numbers, score: f64::MAX }
    }
}


fn eval(numbers: &Numbers, target_sum: i64) -> f64 {
    let the_sum: i64 = numbers.value.to_vec().into_iter().map(|x|x.value).sum();
    let difference: i64 = target_sum - the_sum;
    let absolute_difference = difference.abs();
    return absolute_difference as f64;
}
