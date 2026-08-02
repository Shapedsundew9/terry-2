use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rust_2::genetic_code::GeneticCodeTsetlin;

fn benchmark_tsetlin(c: &mut Criterion) {
    let code = GeneticCodeTsetlin::new(6, 4, 13, 42).unwrap();
    c.bench_function("tsetlin_lookup_6x4", |b| {
        let mut input = 0u64;
        b.iter(|| {
            input = input.wrapping_add(0x9e37) & ((1 << 13) - 1);
            black_box(code.evaluate(black_box(input)))
        });
    });

    let other = GeneticCodeTsetlin::new(6, 4, 13, 43).unwrap();
    c.bench_function("tsetlin_crossover_6x4", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(44);
        b.iter(|| {
            black_box(
                code.crossover_with_rng(&other, black_box(0.01), &mut rng)
                    .unwrap(),
            )
        });
    });

    let wiki_code = GeneticCodeTsetlin::new(16, 16, 24, 45).unwrap();
    c.bench_function("tsetlin_lookup_16x16_wiki", |b| {
        let mut input = 0u64;
        b.iter(|| {
            input = input.wrapping_add(0x9e37) & ((1 << 24) - 1);
            black_box(wiki_code.evaluate(black_box(input)))
        });
    });

    let wiki_other = GeneticCodeTsetlin::new(16, 16, 24, 46).unwrap();
    c.bench_function("tsetlin_crossover_16x16_wiki", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(47);
        b.iter(|| {
            black_box(
                wiki_code
                    .crossover_with_rng(&wiki_other, black_box(0.01), &mut rng)
                    .unwrap(),
            )
        });
    });
}

criterion_group!(benches, benchmark_tsetlin);
criterion_main!(benches);
