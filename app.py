import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Dict
import pandas as pd
import io
from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer


# Load environment variables
load_dotenv()

# Configure the generative AI model
# Ensure you have a GOOGLE_API_KEY set in your environment or .env file
# Using GEMINI_API_KEY as specified in your code
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Google API Key not found.")
    st.error("Please set the GEMINI_API_KEY environment variable or create a .env file.")
    st.stop() # Stop the app if the API key is not found

try:
    genai.configure(api_key=api_key)
    # Using a suitable model for text generation
    # Check model availability and suitability for your needs (cost, performance)
    GEMINI_MODEL = os.getenv("GEMINI_MODEL") or 'gemini-1.5-flash' # Or 'gemini-1.0-pro', 'gemini-1.5-pro'
    # Test if the model is available - this might add latency on startup
    # genai.get_model(GEMINI_MODEL)
except Exception as e:
    st.error(f"Failed to configure Gemini API or access model '{GEMINI_MODEL}': {e}")
    st.stop()


def parse_schema_string(schema_str: str) -> List[str]:
    """Parses a comma or space separated schema string into a list of column names."""
    if not schema_str:
        return []
    # Replace commas with spaces and split, then filter out empty strings
    return [col.strip() for col in schema_str.replace(',', ' ').split() if col.strip()]

def generate_initial_sample_data(table_schemas: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Generates a small sample of mock data for the given table schemas using Gemini.
    The data is returned as a dictionary of Pandas DataFrames.
    """
    prompt = """
    You are a database expert. I will provide you with the schemas for several tables.
    I want you to generate a small sample of realistic mock data for these tables, ensuring that referential integrity is maintained between related tables based on column names (e.g., `ORDER_ID` in one table likely relates to `ORDER_ID` in another).
    The data should be diverse and suitable for training a synthetic data model.

    Provide at least 10 rows and no more than 20 rows for each table.

    Output the data for each table as comma-separated values (CSV format).
    Present the data for each table in a separate block, clearly indicating the table name before the data block.
    DO NOT include column headers in the output data.
    DO NOT include any introductory or concluding text, just the labeled data blocks.

    Table Schemas:
    """
    for table_name, columns in table_schemas.items():
        prompt += f"- {table_name}: {', '.join(columns)}\n"

    prompt += """

    Example Output Format (replace with actual data and table names):
    TableName1:
    value1_1,value1_2,value1_3
    value2_1,value2_2,value2_3
    ...

    TableName2:
    valueA_1,valueA_2
    valueB_1,valueB_2
    ...

    Important Considerations:
    -   Data Types: Use appropriate data types (integers, strings, dates, etc.).
    -   Referential Integrity: Ensure foreign key values exist as primary keys in referenced tables.
    -   Realistic Data: Generate plausible data.
    -   Data Diversity: Vary the data.
    -   Dates: Realistic dates, not in the future.
    -   Currency, Phone Numbers, Emails, Postcodes etc.: Generate realistic values if applicable.
    -   ONLY provide the raw data blocks formatted as specified above.
    """

    # Removed st.info calls from core logic for better testability,
    # but keeping them in the main Streamlit function
    try:
        model = genai.GenerativeModel(model_name=GEMINI_MODEL)
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        if not response_text:
             # Returning an empty dictionary or DataFrames might be better than None
             # for downstream functions, depending on how they handle empty input.
             # Let's return empty DataFrames with correct columns.
             empty_data = {}
             for table_name, cols in table_schemas.items():
                 empty_data[table_name] = pd.DataFrame(columns=cols)
             return empty_data


        # Parse the output
        initial_data = {}
        current_table_name = None
        data_lines = []

        # Split the response by lines and process
        for line in response_text.splitlines():
            line = line.strip()
            if not line: # Skip empty lines
                continue

            # Check if line is a table name indicator (ends with ':')
            if line.endswith(':'):
                # If we were collecting data for a previous table, process it
                if current_table_name and data_lines:
                    try:
                        # Use io.StringIO to treat the list of lines as a file
                        data_io = io.StringIO("\n".join(data_lines))
                        # Read as CSV, no header, specify column names from the schema
                        # Ensure the table name exists in the schema dictionary before accessing
                        if current_table_name in table_schemas:
                            df = pd.read_csv(data_io, header=None, names=table_schemas[current_table_name])
                            initial_data[current_table_name] = df
                        else:
                             # Warning: Skipping data for unexpected table '{current_table_name}' returned by Gemini.
                             pass # In testable function, avoid Streamlit calls


                    except Exception as parse_error:
                         # Error parsing data for table '{current_table_name}': {parse_error}
                         # Attempt to continue with other tables, add an empty df for this one
                         if current_table_name in table_schemas:
                             initial_data[current_table_name] = pd.DataFrame(columns=table_schemas[current_table_name])
                         pass # In testable function, avoid Streamlit calls

                    data_lines = [] # Reset data lines

                # Set the new current table name
                current_table_name = line[:-1].strip()
                # Ensure the table name returned by Gemini is one we requested
                if current_table_name not in table_schemas:
                    # Warning: Gemini returned data for unexpected table: '{current_table_name}'. Skipping.
                    current_table_name = None # Ignore data lines until a known table name is found
                    pass # In testable function, avoid Streamlit calls
            elif current_table_name:
                 # Add the line to the current table's data if we are inside a known table block
                 data_lines.append(line)


        # Process the last table's data if any lines were collected
        if current_table_name and data_lines:
             try:
                data_io = io.StringIO("\n".join(data_lines))
                # Ensure the table name exists in the schema dictionary before accessing
                if current_table_name in table_schemas:
                    df = pd.read_csv(data_io, header=None, names=table_schemas[current_table_name])
                    initial_data[current_table_name] = df
                else:
                     # Warning: Skipping data for unexpected table '{current_table_name}' returned by Gemini.
                     pass # In testable function, avoid Streamlit calls

             except Exception as parse_error:
                 # Error parsing data for table '{current_table_name}': {parse_error}
                 if current_table_name in table_schemas:
                     initial_data[current_table_name] = pd.DataFrame(columns=table_schemas[current_table_name]) # Add an empty df
                 pass # In testable function, avoid Streamlit calls


        # Check if data was generated for all requested tables and add empty DFs for missing ones
        for table_name in table_schemas:
            # Check if table_name is in initial_data keys and if the dataframe is empty
            if table_name not in initial_data or initial_data[table_name].empty:
                 # Warning: No valid data returned for table '{table_name}'. SDV might have issues.
                 # Ensure an empty dataframe with correct columns exists
                 initial_data[table_name] = pd.DataFrame(columns=table_schemas[table_name])
                 pass # In testable function, avoid Streamlit calls


        return initial_data

    except Exception as e:
        # Error calling Gemini API: {e}
        # Return empty dataframes on error
        empty_data = {}
        for table_name, cols in table_schemas.items():
            empty_data[table_name] = pd.DataFrame(columns=cols)
        return empty_data


def synthesize_data_with_sdv(initial_data: Dict[str, pd.DataFrame], scale: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Uses SDV's HMASynthesizer to generate synthetic data based on the initial sample.
    Infers metadata and relationships from the initial data.
    """
    # Check if initial data is valid and not all dataframes are empty
    if not initial_data or all(df.empty for df in initial_data.values()):
        # Warning: No initial data provided or all tables are empty. Cannot synthesize data.
        return {} # Return empty dictionary if no valid data


    try:
        # Removed st.info calls from core logic for better testability
        # Use the correct class method to detect metadata from dataframes
        metadata = MultiTableMetadata()
        metadata.detect_from_dataframes(data=initial_data)

        # Optional: Print inferred metadata for verification in test logs
        # print("\nInferred Metadata:")
        # print(metadata.to_dict())

        # Training SDV HMA Synthesizer...
        synthesizer = HMASynthesizer(metadata)
        synthesizer.fit(initial_data)

        # Generating synthetic data...
        synthetic_data = synthesizer.sample(scale=scale)

        return synthetic_data

    except Exception as e:
        # Error during SDV synthesis: {e}
        return {} # Return empty dictionary on error


def main():
    st.title("Synthetic Data Generator (Gemini + SDV)")

    st.write("Enter your table schemas below. Use spaces or commas to separate column names.")
    st.write("The tool will use this to generate a small sample with Gemini, infer relationships with SDV, and then synthesize a larger dataset.")

    # Number of tables input
    num_tables = st.number_input("Number of Tables:", min_value=1, max_value=5, value=1, step=1)

    # Table schemas input - use a list to maintain order
    table_inputs = []
    # Dictionary to store parsed schemas (table_name -> list of columns)
    table_schemas_dict: Dict[str, List[str]] = {}

    # Use columns for better layout of name and schema inputs
    for i in range(num_tables):
        col1, col2 = st.columns([1, 3])
        with col1:
            table_name = st.text_input(f"Table {i + 1} Name:", value=f"Table_{i+1}", key=f"table_name_{i}")
        with col2:
            table_schema_str = st.text_area(f"Table {i + 1} Schema (comma or space separated columns):",
                                          placeholder="e.g., id, name, email, created_at", key=f"table_schema_{i}")
        # Store input for processing
        table_inputs.append({"name": table_name.strip(), "schema_str": table_schema_str.strip()})

    # Populate the schema dictionary only with valid inputs
    for table_input in table_inputs:
        if table_input["name"] and table_input["schema_str"]:
            columns = parse_schema_string(table_input["schema_str"])
            if columns: # Only add if columns were successfully parsed
                 table_schemas_dict[table_input["name"]] = columns
            else:
                 st.warning(f"No valid columns found for table '{table_input['name']}'. Please check the schema format.")


    # Scale factor for SDV synthesis
    # Disable the slider if no valid schemas are entered
    scale_factor = st.slider("Synthetic Data Scale Factor (compared to initial sample):",
                             min_value=1, max_value=100, value=10,
                             disabled=not bool(table_schemas_dict))

    # Generate data button
    # Disable button if no valid schemas are entered for any table
    if st.button("Generate Synthetic Data", disabled=not bool(table_schemas_dict)):
        if not table_schemas_dict:
             st.error("Please provide names and schemas for at least one table with valid columns.")
             return

        # Step 1: Generate initial sample data using Gemini
        st.info("Generating initial sample data using Gemini...") # Moved st.info here
        initial_sample_data = generate_initial_sample_data(table_schemas_dict)

        # Check if initial data generation was successful and returned data for requested tables
        # Note: The check `df.columns.tolist() == table_schemas_dict.get(df.columns.name, [])` might be too strict
        # if Gemini returns data with slightly different column names or ordering.
        # A simpler check might be if the dataframe is not empty for expected tables.
        # Let's rely on the checks within generate_initial_sample_data to return empty DFs for issues.
        if not initial_sample_data or all(df.empty for df in initial_sample_data.values()):
             st.error("Failed to generate sufficient initial sample data with Gemini. Cannot proceed with SDV synthesis. Please check the schema or try again.")
             # Display any partial data generated for debugging
             if initial_sample_data:
                 st.subheader("Partial Initial Sample Data (for debugging)")
                 for name, df in initial_sample_data.items():
                     st.write(f"**{name} ({len(df)} rows):**")
                     st.dataframe(df)
             return


        st.subheader("Initial Sample Data (Generated by Gemini)")
        # Display initial dataframes
        for name, df in initial_sample_data.items():
            st.write(f"**{name} ({len(df)} rows):**")
            st.dataframe(df)

        # Step 2: Synthesize larger dataset using SDV
        st.info("Starting SDV synthesis...") # Moved st.info here
        synthetic_data = synthesize_data_with_sdv(initial_sample_data, scale=scale_factor)

        if synthetic_data:
            st.subheader("Synthetic Data (Generated by SDV)")
            # Display synthetic dataframes and provide download buttons
            for name, df in synthetic_data.items():
                st.write(f"**{name} ({len(df)} rows):**")
                st.dataframe(df)
                # Option to download data as CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {name} Data as CSV",
                    data=csv,
                    file_name=f'synthetic_{name.lower()}.csv',
                    mime='text/csv',
                    key=f'download_{name}' # Unique key for each download button
                )
        else:
            st.error("Failed to generate synthetic data using SDV.")

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
