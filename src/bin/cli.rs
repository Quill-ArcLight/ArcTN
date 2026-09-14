use std::str::FromStr;

pub fn value<T: FromStr>(args: &[String], index: usize) -> T {
    let flag = &args[index];
    let raw = args
        .get(index + 1)
        .filter(|raw| !raw.starts_with("--"))
        .unwrap_or_else(|| {
            eprintln!("{flag} requires a value");
            std::process::exit(2);
        });
    raw.parse().unwrap_or_else(|_| {
        eprintln!("invalid value for {flag}: {raw:?}");
        std::process::exit(2);
    })
}
