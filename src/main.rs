use std::f32::consts::TAU;

struct AdaMax {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    m: f32,
    u: f32,
    t: f32
}

impl AdaMax {
    fn new(alpha: f32) -> Self {
        Self {
            alpha,
            beta1: 0.9,
            beta2: 0.999,
            m: 0.0,
            u: 0.0,
            t: 0.0
        }
    }

    fn update(&mut self, param: f32, gradient: f32) -> f32 {
        const EPSILON: f32 = 0.000001;
        self.t += 1.0;
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient;
        self.u = f32::max(self.beta2 * self.u, gradient.abs());
        param - (self.alpha * self.m)/((1.0 - self.beta1.powf(self.t)) * (self.u + EPSILON))
    }
}

struct RNG {
    seed: u32,
    pos: u32
}

impl RNG {
    fn new(seed: u32) -> Self { Self {seed, pos: 1} }
    
    fn sample(&mut self) -> f32 {
        const NOISE1: u32 = 0x68E31DA4;
        const NOISE2: u32 = 0xB5297A4D;
        const NOISE3: u32 = 0x1B56C4E9;
        
        let mut mangled = self.pos as u32;
        mangled = mangled.wrapping_mul(NOISE1);
        mangled = mangled.wrapping_add(self.seed);
        mangled ^= mangled >> 8;
        mangled = mangled.wrapping_add(NOISE2);
        mangled ^= mangled << 8;
        mangled = mangled.wrapping_mul(NOISE3);
        mangled ^= mangled >> 8;
        self.pos += 1;
        (mangled as f32) / (u32::MAX as f32)
    }
}

fn normal_sample(rng: &mut RNG, mean: f32, stddev: f32) -> f32 {
    let u1: f32 = rng.sample();
    let u2: f32 = rng.sample();
    let mag: f32 = stddev * (-2.0 * u1.ln()).sqrt();
    mag * (TAU * u2).cos() + mean
}

fn normal_pdf(x: f32, mean: f32, stddev: f32) -> f32 {
    const HALF_LN_TAU: f32 = 0.91893853320467274178032973640562;
    let z = (x - mean) / stddev;
    -stddev.ln() - HALF_LN_TAU - 0.5 * z.powf(2.0)
}

fn main() {
    let mut rng = RNG::new(123);

    let guess = 8.5;
    let measurement = 9.5;
    let mut a = guess;
    let mut b = 1.0;    
    
    let npop = 50;
    let sigma = 0.995;
    let mut optim = AdaMax::new(0.1);
    let mut data_log: Vec<(i32, f32)> = Vec::new();
    
    for i in 0..2500 {
        let mut returns_a = 0.0;
        let mut returns_b = 0.0;
        let mut sum_elbo = 0.0;
        for x in 0..npop {
            let a_eps = normal_sample(&mut rng, 0.0, 1.0);
            let b_eps = normal_sample(&mut rng, 0.0, 1.0);
            let a_t = a + sigma * a_eps;
            let b_t = b + sigma * b_eps;
            let stddev = (0.5 * b_t).exp();
            let weight_given_measurement = normal_sample(&mut rng, a_t, stddev);
            let q = normal_pdf(weight_given_measurement, a_t, stddev);
            let p = normal_pdf(measurement, weight_given_measurement, 0.75) + normal_pdf(weight_given_measurement, guess, 1.0);
            let elbo = q - p;
            returns_a = returns_a + elbo * a_eps;
            returns_b = returns_b + elbo * b_eps;
            sum_elbo = sum_elbo + elbo;
        }
        a = optim.update(a, returns_a/(npop as f32 * sigma));
        b = optim.update(b, returns_b/(npop as f32 * sigma));
        data_log.push((i, sum_elbo / npop as f32));
        println!("Generation: {}, ELBO: {}, mu: {}, sigma: {}", i, sum_elbo / npop as f32, a, (0.5 * b).exp());
    }
}