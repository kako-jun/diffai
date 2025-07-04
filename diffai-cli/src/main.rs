use clap::Parser;
use diffai_core::diff_data;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// First data source
    #[arg(short = '1', long)]
    data1: String,

    /// Second data source
    #[arg(short = '2', long)]
    data2: String,
}

fn main() {
    let args = Args::parse();

    println!("Diffing AI/ML data...");
    let result = diff_data(&args.data1, &args.data2);
    println!("{}", result);
}
