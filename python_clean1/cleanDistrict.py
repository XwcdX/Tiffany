import os
import pandas as pd

def split_and_clean_districts(
    input_path: str,
    output_with: str = "with_district.xlsx",
    # output_without: str = "no_district.xlsx"
):
    df = pd.read_excel(input_path)

    mask_not_found = df["District"].astype(str) == "Not Found"
    # df_no = df[mask_not_found].copy()
    df_yes = df[~mask_not_found].copy()

    df_yes["District"] = df_yes["District"].astype(str).str.replace(
        r"^Kecamatan\s+", "", regex=True
    )

    df_yes.to_excel(output_with, index=False)
    # df_no.to_excel(output_without, index=False)

    print(f"Wrote {len(df_yes)} rows to '{output_with}'")
    # print(f"Wrote {len(df_no)} rows to '{output_without}'")

if __name__ == "__main__":
    INPUT = "no_redundant_data.xlsx"
    base = os.path.dirname(INPUT)
    OUT_WITH = os.path.join(base, "has_address_valid_district3.xlsx")
    # OUT_NO   = os.path.join(base, "has_address_no_district2.xlsx")

    split_and_clean_districts(INPUT, OUT_WITH)
