// Import the necessary crates and modules
use clap::Parser;
use csv::ReaderBuilder;
use indoc::formatdoc;
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{create_dir_all, write},
    path::PathBuf,
};

// Define the command line arguments using the clap crate
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Open images Directory to use
    #[arg(short, long)]
    input_file: PathBuf,

    /// Output Directory
    #[arg(short, long)]
    output_dir: PathBuf,
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
    // Parse the command line arguments
    let cli = Cli::parse();

    // Read the CSV file using the csv crate
    let mut csv_file = ReaderBuilder::new().from_path(&cli.input_file).unwrap();

    // Create a HashMap to store the object detections by file name
    let mut file_map: HashMap<String, Vec<Detection>> = HashMap::new();

    // Iterate through the records in the CSV file and add them to the HashMap
    for record in csv_file.deserialize() {
        let detect: Detection = record.unwrap();
        if file_map.contains_key(&detect.filename) {
            file_map.get_mut(&detect.filename).unwrap().push(detect);
        } else {
            file_map
                .insert(detect.filename.clone(), vec![detect]);
        }
    }
    create_dir_all(&cli.output_dir).unwrap();

    // Get the input directory path as a string
    let folder = cli.input_file.parent().unwrap().to_str().unwrap();

    // Iterate through the values of the file_map HashMap in parallel and convert each file to Pascal VOC format
    file_map.values_mut().par_bridge().for_each(|file| {
        let file_string = to_voc_string(file, folder);
        let mut file_path = cli.output_dir.clone().join(&file[0].filename);
        file_path.set_extension("xml");
        write(file_path, file_string).unwrap();
    });
}

/// Convert a vector of Detection structs to a Pascal VOC XML string
fn to_voc_string(file: &[Detection], folder: &str) -> String {
    let filename = &file[0].filename;
    let objects: String = file
        .par_iter()
        .map(|detections| {
            formatdoc! {
                "<object>
                        <name>{}</name>
                        <pose>Unspecified</pose>
                        <truncated>1</truncated>
                        <difficult>0</difficult>
                        <bndbox>
                            <xmin>{}</xmin>
                            <ymin>{}</ymin>
                            <xmax>{}</xmax>
                            <ymax>{}</ymax>
                        </bndbox>
                </object>\n",
                detections.class, detections.xmin, detections.ymin, detections.xmax, detections.ymax

            }
        })
        .collect();
    formatdoc! {
        "<annotation verified=\"yes\">
            <folder>{folder}</folder>
            <filename>{filename}</filename>
            <path>{path}</path>
            <source>
                <database>Unknown</database>
            </source>
            <size>
                <width>{width}</width>
                <height>{height}</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
            {objects}
        </annotation>",
        path = format!("{folder}/{filename}"),
        width = file[0].width,
        height = file[0].height
    }
}

#[test]
fn test_to_voc_string() {
    // Define input
    let detections = vec![
        Detection {
            filename: "image1.jpg".to_string(),
            width: 640,
            height: 480,
            class: "car".to_string(),
            xmin: 50,
            ymin: 100,
            xmax: 200,
            ymax: 250,
        },
        Detection {
            filename: "image1.jpg".to_string(),
            width: 640,
            height: 480,
            class: "truck".to_string(),
            xmin: 300,
            ymin: 150,
            xmax: 500,
            ymax: 300,
        },
    ];
    let folder = "/path/to/images";

    // Define expected output
    let expected_output = formatdoc! {"
        <annotation verified=\"yes\">
            <folder>/path/to/images</folder>
            <filename>image1.jpg</filename>
            <path>/path/to/images/image1.jpg</path>
            <source>
                <database>Unknown</database>
            </source>
            <size>
                <width>640</width>
                <height>480</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
            <object>
                <name>car</name>
                <pose>Unspecified</pose>
                <truncated>1</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>50</xmin>
                    <ymin>100</ymin>
                    <xmax>200</xmax>
                    <ymax>250</ymax>
                </bndbox>
        </object>
        <object>
                <name>truck</name>
                <pose>Unspecified</pose>
                <truncated>1</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>300</xmin>
                    <ymin>150</ymin>
                    <xmax>500</xmax>
                    <ymax>300</ymax>
                </bndbox>
        </object>\n
        </annotation>"};

    // Call the function and assert that the output matches the expected output
    let output = to_voc_string(&detections, folder);
    assert_eq!(output, expected_output);
}
