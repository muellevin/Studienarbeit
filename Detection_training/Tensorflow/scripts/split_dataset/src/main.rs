use clap::Parser;
use csv::{ReaderBuilder, Writer};
use serde_derive::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, copy, create_dir_all};
use std::path::{Path, PathBuf};

// Define the command line arguments using the clap crate
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Images Directory to use
    #[arg(short, long)]
    input_dir: PathBuf,

    /// Set Directory
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Output in Trainset
    #[arg(short, long, default_value_t = 0.85)]
    train_percent: f32,

    /// Output in Trainset
    #[arg(short = 'e', long, default_value_t = 0.1)]
    test_percent: f32,

    /// Output in devset
    #[arg(short, long, default_value_t = 0.05)]
    dev_percent: f32,
}

#[derive(Debug, Serialize, Deserialize)]
// A struct representing a converted object detection for output to the CSV file.
struct Detection {
    filename: String,
    width: u16,
    height: u16,
    class: String,
    xmin: u16,
    ymin: u16,
    xmax: u16,
    ymax: u16,
}

fn main() {
    let cli = Cli::parse();
    let mut images = Vec::new();
    if let Ok(entries) = fs::read_dir(&cli.input_dir) {
        for entry in entries.into_iter().flatten() {
            if let Some(file_name) = entry.file_name().to_str() {
                if !file_name.ends_with(".csv") && !file_name.ends_with(".xml") {
                    images.push(file_name.to_string());
                }
            }
        }
    }
    let train_set: HashSet<&String> = HashSet::from_iter(
        images
            .iter()
            .take(((images.len() as f32) * cli.train_percent) as usize),
    );
    let test_set: HashSet<&String> = HashSet::from_iter(
        images
            .iter()
            .skip(((images.len() as f32) * cli.train_percent) as usize)
            .take(((images.len() as f32) * cli.test_percent) as usize),
    );
    let dev_set: HashSet<&String> = HashSet::from_iter(
        images
            .iter()
            .skip(
                ((images.len() as f32) * cli.train_percent) as usize
                    + ((images.len() as f32) * cli.test_percent) as usize,
            )
            .take(((images.len() as f32) * cli.test_percent) as usize + 1),
    );
    let mut records: Vec<Detection> = Vec::new();
    let mut csv_file = ReaderBuilder::new()
        .from_path(&cli.input_dir.join("detections.csv"))
        .unwrap();

    for record in csv_file.deserialize() {
        records.push(record.unwrap());
    }
    write_csv(
        &cli.input_dir as &Path,
        &cli.output_dir.join("trainset") as &Path,
        train_set,
        &mut records,
    );
    write_csv(
        &cli.input_dir as &Path,
        &cli.output_dir.join("testset") as &Path,
        test_set,
        &mut records,
    );
    write_csv(
        &cli.input_dir as &Path,
        &cli.output_dir.join("devset") as &Path,
        dev_set,
        &mut records,
    );
}

/// Write the list of available detections to the 'detections.csv' file in the output directory.
/// And copies the images to the dataset.
///
/// # Arguments
///
/// * `input_dir` - The path to the directory containing the original image files.
/// * `output_dir` - The path to the directory where the image files will be copied to.
/// * `files` - A HashSet of string references representing the filenames of the images that should be processed.
/// * `records` - A mutable slice of Detection objects representing the detections that should be written to the output CSV file.
fn write_csv(
    input_dir: &Path,
    output_dir: &Path,
    files: HashSet<&String>,
    records: &mut [Detection],
) {
    // Delete the output directory if it exists.
    if output_dir.exists() {
        fs::remove_dir_all(output_dir).unwrap();
    }
    create_dir_all(output_dir).unwrap();
    // Create the output CSV file
    let output_csv = output_dir.join("detections.csv");

    // Create a CSV writer for the output file
    let mut wtr = Writer::from_path(&output_csv).unwrap();

    // Iterate over the records and copy the corresponding images and write the records to the output CSV file
    for record in records.iter_mut() {
        // Check if the record's filename is in the set of files that should be processed
        if files.contains(&record.filename) {
            // Copy the image file from the input directory to the output directory
            copy(
                &input_dir.join(&record.filename),
                output_dir.join(&record.filename),
            )
            .unwrap();
            // Write the record to the output CSV file
            wtr.serialize(record).unwrap();
        }
    }

    // Print the number of records that were written to the CSV file
    println!("Wrote {:?} records in {:?}", files.len(), output_csv);

    // Flush the CSV writer to ensure that all changes have been written to the file
    wtr.flush().unwrap();
}
