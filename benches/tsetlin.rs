use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rust_2::genetic_code::GeneticCodeTsetlin;

fn benchmark_tsetlin(c: &mut Criterion) {
    let code = GeneticCodeTsetlin::new(6, 10, 13, None, 42).unwrap();
    c.bench_function("tsetlin_lookup_6x10", |b| {
        let mut input = 0u64;
        b.iter(|| {
            input = input.wrapping_add(0x9e37) & ((1 << 13) - 1);
            black_box(code.evaluate(black_box(input)))
        });
    });

    let other = GeneticCodeTsetlin::new(6, 10, 13, None, 43).unwrap();
    c.bench_function("tsetlin_crossover_6x10", |b| {
        let mut rng = StdRng::seed_from_u64(44);
        b.iter(|| {
            black_box(
                code.crossover_with_rng(&other, black_box(0.01), &mut rng)
                    .unwrap(),
            )
        });
    });
}

criterion_group!(benches, benchmark_tsetlin);
criterion_main!(benches);
