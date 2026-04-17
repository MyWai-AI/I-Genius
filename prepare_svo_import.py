import os
import csv
import zipfile
import time
import argparse

def generate_mywai_import_zip(svo_dir, output_zip="mywai_import.zip", serial_number="VILMA_ROBOT_01", camera_sensor_name="CAMERA1"):
    frames_dir = os.path.join(svo_dir, "frames", "myvol2")
    frames_index_path = os.path.join(frames_dir, "frames_index.csv")
    
    if not os.path.exists(frames_index_path):
        print(f"Error: {frames_index_path} not found.")
        return

    # Base timestamp (current time in milliseconds) to ensure events are recorded recently
    base_timestamp_ms = int(time.time() * 1000)
    
    events_csv_path = os.path.join(svo_dir, "events.csv")
    
    # Read the frames index and generate events.csv
    with open(frames_index_path, 'r', encoding='utf-8') as infile, \
         open(events_csv_path, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile, delimiter=';')
        
        # Header according to MYWAI Data Import documentation
        writer.writerow(["TIMESTAMP", "SERIAL_NUMBER", f"<{camera_sensor_name}>"])
        
        for row in reader:
            # time_sec to milliseconds, added to base_timestamp
            time_sec = float(row['time_sec'])
            timestamp_ms = base_timestamp_ms + int(time_sec * 1000)
            
            filename = row['filename']
            # We assume the image filename goes in the camera column
            writer.writerow([timestamp_ms, serial_number, filename])

    print(f"Generated {events_csv_path}")

    # Now create the ZIP file including events.csv and all images
    print(f"Creating {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add events.csv
        zipf.write(events_csv_path, arcname="events.csv")
        
        # Add all images from the frames dir
        for filename in os.listdir(frames_dir):
            if filename.endswith(".jpg"):
                filepath = os.path.join(frames_dir, filename)
                # The images should be at the root of the zip alongside events.csv usually
                zipf.write(filepath, arcname=filename)
                
    print(f"Successfully created {output_zip}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package SVOfile into MYWAI Data Import format")
    parser.add_argument("--svo-dir", default="SVOfile", help="Path to the extracted SVOfile directory")
    parser.add_argument("--output", default="SVOfile_import.zip", help="Output ZIP path")
    parser.add_argument("--serial", default="VILMA_ROBOT_01", help="Equipment Serial Number")
    parser.add_argument("--camera-sensor", default="CAMERA1", help="Camera Sensor Name on MYWAI")
    
    args = parser.parse_args()
    
    generate_mywai_import_zip(args.svo_dir, args.output, args.serial, args.camera_sensor)
