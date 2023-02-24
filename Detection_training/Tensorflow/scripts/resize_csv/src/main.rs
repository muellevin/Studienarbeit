use clap::Parser;
use csv::{ReaderBuilder, Writer};
use std::{path::{PathBuf}, fs::{self, create_dir_all}};
use serde_derive::{Deserialize, Serialize};

// Define the command line arguments using the clap crate
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Open images Directory to use
    #[arg(short, long)]
    input_file: PathBuf,

    /// Output Directory
    #[arg(short, long)]
    output_file: PathBuf,

    /// Output Directory
    #[arg(short, long)]
    resize_height: u16,
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

impl Detection {
    fn resize_height(&mut self, new_height: u16){
        let img_ratio = self.width/self.height;
        let new_width = new_height * img_ratio;
        let width_ratio = new_height/self.width;
        let height_ratio = new_height/self.height;

        self.ymin *= width_ratio;
        self.ymax *= width_ratio;
        self.xmin *= height_ratio;
        self.xmax *= height_ratio;

        self.width = new_width;
        self.height = new_height;
    }
}


fn main() {
    let cli = Cli::parse();
    let mut csv_file = ReaderBuilder::new()
    .from_path(cli.input_file)
    .unwrap();
    let mut records: Vec<Detection> = Vec::new();
    println!("starting to resize detection in csv file");
    for record in csv_file.deserialize() {
        let mut detect: Detection = record.unwrap();
        detect.resize_height(cli.resize_height);
        records.push(detect);
    }
    // Cleaning output Directory from all files
    // Delete the output directory if it exists.
    let output_dir = cli.output_file.parent().unwrap();
    if output_dir.exists() {
        fs::remove_dir_all(output_dir).unwrap();
    }
    create_dir_all(output_dir).unwrap();
    // Write the list of available detections to the 'detections.csv' file in the output directory. If it exists it will join them.
    println!("Writing output file");
    let mut wtr = Writer::from_path(cli.output_file).unwrap();
    for record in records.iter_mut() {
        wtr.serialize(record).unwrap();
    }
    // Flush the buffer to write the changes to the file
    wtr.flush().unwrap();
}