use clap::Parser;
use csv::{ReaderBuilder, Writer};
use imagesize::size;
use serde_derive::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{copy, create_dir_all};
use std::path::{Path, PathBuf};

// Define the command line arguments using the clap crate
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Open images Directory to use
    #[arg(short, long)]
    input_dir: PathBuf,

    /// Output Directory
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Labels which should be copied
    #[arg(short, long)]
    labels: Vec<String>,
}

#[allow(non_snake_case)]
/// Allow non-snake case identifiers in this struct for deserialization
#[derive(Debug, Deserialize)]
/// A struct representing a single object detection from the input CSV file.
struct Detection {
    ImageID: String,
    LabelName: String,
    XMin: f32,
    XMax: f32,
    YMin: f32,
    YMax: f32,
}

#[derive(Debug, Serialize, Deserialize)]
// A struct representing a converted object detection for output to the CSV file.
struct ConvertedDetection {
    filename: String,
    width: u16,
    height: u16,
    class: String,
    xmin: u16,
    ymin: u16,
    xmax: u16,
    ymax: u16,
}

impl ConvertedDetection {
    /// A constructor method that converts a Detection struct to a ConvertedDetection struct.
    fn from_detection(detection: &Detection, width: u16, height: u16, class: &str) -> Self {
        ConvertedDetection {
            filename: detection.ImageID.to_owned(),
            width,
            height,
            class: class.to_string(),
            xmin: (detection.XMin * width as f32) as u16,
            ymin: (detection.YMin * height as f32) as u16,
            xmax: (detection.XMax * width as f32) as u16,
            ymax: (detection.YMax * height as f32) as u16,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let allowed_labels = get_allowed_labels(&cli.labels, &cli.input_dir as &Path);

    delete_unused_labels(
        &allowed_labels,
        &cli.input_dir as &Path,
        &cli.output_dir as &Path,
    );
}

/// This function takes in a reference to a vector of strings `labels` and a reference to a `Path` `path`,
/// and returns a `HashMap` of allowed labels.
fn get_allowed_labels(labels: &[String], path: &Path) -> HashMap<String, String> {
    // Create a new empty `HashMap` named `allowed_labels`.
    let mut allowed_labels = HashMap::new();

    // Open the CSV file containing the classes metadata using the `csv` crate and the provided `path`.
    let mut csv_file = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path.join("metadata/classes.csv"))
        .unwrap();
    println!("Obtaining classes from: {:#?}", &path);
    // Iterate over each record in the CSV file using the `deserialize` method of the `csv` crate.
    for result in csv_file.deserialize() {
        // Parse each record into a tuple of `(String, String)` using `unwrap`.
        let record: (String, String) = result.unwrap();
        // Check if the second element of the tuple (the label) is contained in the `labels` vector.
        if labels.contains(&record.1) {
            println!("Found allowed label: {:?}", &record.1);
            // If the label is allowed, insert the tuple into the `allowed_labels` `HashMap`.
            allowed_labels.insert(record.0, record.1);
        }
    }
    // Return the `allowed_labels` `HashMap`.
    allowed_labels
}

/// This function takes in the list of allowed labels, the input directory path and output directory path.
///
/// It reads the 'detections.csv' file from the input directory and iterates through each record.
/// If the LabelName of the record is found in the list of allowed labels, it adds the converted record to a vector.
///
/// The converted record is created using the ConvertedDetection struct and the from_detection method.
/// The from_detection method calculates the new coordinates based on the image size and adds the new record to the vector.
///
/// It then creates the destination directory if it does not exist and copies the corresponding image to the output directory.
///
/// Finally, it writes the list of available detections to the 'detections.csv' file in the output directory.
fn delete_unused_labels(
    allowed_labels: &HashMap<String, String>,
    input_dir: &Path,
    output_dir: &Path,
) {
    // Open the 'detections.csv' file
    let mut csv_file = ReaderBuilder::new()
        .from_path(input_dir.join("labels/detections.csv"))
        .unwrap();
    println!("Obtaining detections from: {:?}", &input_dir);

    let mut available_detections: Vec<ConvertedDetection> = Vec::new();

    // Create the destination directory if it does not exist
    if !output_dir.exists() {
        create_dir_all(output_dir).unwrap();
    }
    // Iterate through each record in the 'detections.csv' file
    for result in csv_file.deserialize() {
        let mut record: Detection = result.unwrap();
        // If the LabelName of the record is found in the list of allowed labels
        if allowed_labels.contains_key(&record.LabelName) {
            // Append '.jpg' to the ImageID to get the image file name
            record.ImageID.push_str(".jpg");
            let mut image_path = input_dir.join("data");
            image_path = image_path.join(&record.ImageID);

            let output_file = output_dir.join(&record.ImageID);
            // Copy the image file to the output directory
            if !output_file.exists() {
                copy(&image_path, output_file).unwrap();
            }
            // Get the size of the image and convert the detection coordinates
            match size(image_path) {
                Ok(dim) => {
                    let converted = ConvertedDetection::from_detection(
                        &record,
                        dim.width as u16,
                        dim.height as u16,
                        allowed_labels.get(&record.LabelName).unwrap(),
                    );
                    available_detections.push(converted);
                }
                Err(why) => println!("Error getting size: {:?}", why),
            }
        }
    }
    println!("Found {:?} valid detections", available_detections.len());
    // Write the csv file
    write_csv(output_dir, &mut available_detections);
}

/// Write the list of available detections to the 'detections.csv' file in the output directory. If it exists it will join them.
fn write_csv(output_dir: &Path, records: &mut Vec<ConvertedDetection>) {
    let output_csv = output_dir.join("detections.csv");
    let new_records = records.len();
    // check if the output file already exists and joins it
    if output_csv.exists() {
        println!("Found existing output file. Joining it now.");
        let mut csv_file = ReaderBuilder::new().from_path(&output_csv).unwrap();
        for record in csv_file.deserialize() {
            records.push(record.unwrap());
        }
    }
    let mut wtr = Writer::from_path(output_csv).unwrap();
    for record in records.iter_mut() {
        wtr.serialize(record).unwrap();
    }
    println!("Wrote {:?} new records. All detections count: {:?}", new_records, records.len());
    // Flush the buffer to write the changes to the file
    wtr.flush().unwrap();
}
