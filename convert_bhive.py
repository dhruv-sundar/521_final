import csv
import subprocess
import os
import multiprocessing
import tempfile
import glob
import shutil
from functools import partial

def run_tokenizer(hex_code):
    """Run the tokenizer binary on the given hex code and return the output."""
    try:
        result = subprocess.run(
            ["data_collection/build/bin/tokenizer", hex_code, "--token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running tokenizer on {hex_code}: {e}")
        return ""

def process_chunk(chunk_data, temp_dir, chunk_id):
    """Process a chunk of data and write to a temporary file."""
    temp_file = os.path.join(temp_dir, f"temp_output_{chunk_id}.csv")
    
    with open(temp_file, 'w') as outfile:
        # Process each line in the chunk
        start_id = chunk_data[0][0]  # First ID in this chunk
        
        for id_offset, (_, hex_code, timing) in enumerate(chunk_data):
            code_id = start_id + id_offset
            
            # Run tokenizer to get code_xml
            code_xml = run_tokenizer(hex_code)
            
            # Write to output file
            outfile.write(f"{code_id},{timing},,{code_xml}\n")
            
            # Print progress occasionally
            if id_offset % 10 == 0:
                print(f"Chunk {chunk_id}: Processed {id_offset}/{len(chunk_data)} entries")
    
    return temp_file

def convert_bhive_csv(input_file, output_file, num_processes=None):
    """Convert the bhive CSV to the desired format using multiprocessing."""
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Read all data and assign IDs
        all_data = []
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile)
            for i, row in enumerate(reader, 1):
                if len(row) >= 2:
                    hex_code = row[0]
                    timing = row[1]
                    all_data.append((i, hex_code, timing))
        
        # Split data into chunks
        chunk_size = len(all_data) // num_processes
        if chunk_size == 0:
            chunk_size = 1
        
        chunks = []
        for i in range(0, len(all_data), chunk_size):
            chunks.append(all_data[i:i + chunk_size])
        
        print(f"Processing {len(all_data)} entries with {num_processes} processes in {len(chunks)} chunks")
        
        # Process chunks in parallel - FIX: Create arguments list with temp_dir already included
        args = [(chunks[i], temp_dir, i) for i in range(len(chunks))]
        with multiprocessing.Pool(processes=num_processes) as pool:
            temp_files = pool.starmap(process_chunk, args)
        
        # Concatenate temporary files
        with open(output_file, 'w') as outfile:
            # Write header
            outfile.write("code_id,timing,code_intel,code_xml\n")
            
            # Concatenate all temp files
            for temp_file in temp_files:
                with open(temp_file, 'r') as infile:
                    shutil.copyfileobj(infile, outfile)
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    input_file = "bhive_haswell.csv"
    output_file = "converted_bhive.csv"
    
    print(f"Converting {input_file} to {output_file}...")
    convert_bhive_csv(input_file, output_file)
    print("Conversion complete!")