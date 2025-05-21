import os
import pandas as pd
import numpy as np # For np.nan if you need to explicitly handle it

def split_and_clean_districts(
    input_path: str,
    output_with: str = "with_district.xlsx",
):
    """
    Reads an Excel file, filters rows based on the 'JenisKulit' column,
    cleans the 'JenisKulit' column, and saves the result.

    Rows are kept if 'JenisKulit' is not NaN, None, an empty string,
    or a string containing only whitespace.
    For kept rows, "Kecamatan " prefix (and surrounding whitespace) is removed
    from 'JenisKulit'.
    """
    try:
        df = pd.read_excel(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return
    except Exception as e:
        print(f"Error reading Excel file '{input_path}': {e}")
        return

    if "JenisKulit" not in df.columns:
        print(f"Error: Column 'JenisKulit' not found in the input file '{input_path}'.")
        return

    df_filtered = df.copy()

    df_filtered.dropna(subset=["JenisKulit"], inplace=True)
    
    mask_is_actual_string_and_not_empty = df_filtered["JenisKulit"].astype(str).str.strip() != ""
    df_yes = df_filtered[mask_is_actual_string_and_not_empty].copy()

    # --- End of removal logic ---
    if df_yes.empty:
        print(f"No rows with valid 'JenisKulit' data found after filtering from '{input_path}'.")
        try:
            pd.DataFrame().to_excel(output_with, index=False)
            print(f"Wrote 0 rows to '{output_with}' as no valid 'JenisKulit' data was found.")
        except Exception as e:
            print(f"Error writing empty Excel file '{output_with}': {e}")
        return


    try:
        df_yes.to_excel(output_with, index=False)
        print(f"Wrote {len(df_yes)} rows to '{output_with}'")
    except Exception as e:
        print(f"Error writing Excel file '{output_with}': {e}")


if __name__ == "__main__":
    INPUT = "no_redundant_data.xlsx"  # Your specified input file

    # Check if the input file exists before proceeding
    if not os.path.exists(INPUT):
        print(f"Error: Input file '{INPUT}' not found. Please ensure it exists in the same directory or provide the full path.")
    else:
        base = os.path.dirname(os.path.abspath(INPUT)) # Get absolute path's directory
        OUT_WITH = os.path.join(base, "has_address_valid_skin_type.xlsx")
        # OUT_NO   = os.path.join(base, "has_address_no_district2.xlsx") # Still commented

        split_and_clean_districts(INPUT, OUT_WITH)

        # Optional: Verify output if you want to quickly check
        if os.path.exists(OUT_WITH):
            try:
                df_out = pd.read_excel(OUT_WITH)
                print(f"\nSuccessfully processed. Output file '{OUT_WITH}' contains {len(df_out)} rows.")
                if not df_out.empty:
                    print("First 5 rows of the output:")
                    print(df_out.head())
                else:
                    print("The output file is empty, which might be expected if all 'JenisKulit' were invalid.")
            except Exception as e:
                print(f"Could not read or verify the output file '{OUT_WITH}': {e}")
        else:
            print(f"\nOutput file '{OUT_WITH}' was not created. Check previous messages for errors.")