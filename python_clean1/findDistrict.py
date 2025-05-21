import os
import requests
import pandas as pd
from dotenv import load_dotenv

def get_district(address: str, api_key: str) -> str:
    """
    Return the 'administrative_area_level_3' (district) for the given address,
    or "Not Found" on any error / if not present.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    r = requests.get(url, params=params, timeout=10)
    j = r.json()
    if j.get("status") != "OK":
        return "Not Found"
    for comp in j["results"][0].get("address_components", []):
        if "administrative_area_level_3" in comp["types"]:
            return comp["long_name"]
    return "Not Found"

def main():
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise RuntimeError("Please set GOOGLE_API_KEY in your environment.")

    df = pd.read_excel("has_address2.xlsx")

    df_empty = df[df["Address"].isna() | (df["Address"].astype(str).str.strip() == "")]
    df_to_geocode = df.drop(df_empty.index)

    df_to_geocode["District"] = df_to_geocode["Address"].astype(str).apply(
        lambda a: get_district(a, API_KEY)
    )

    df_final = pd.concat([
        df_to_geocode,
        df_empty.assign(District=pd.NA)
    ]).sort_index()

    df_final.to_excel("has_address_with_district2.xlsx", index=False)
    df_empty.to_excel("has_address_empty_address2.xlsx", index=False)

    print("✅ Done.")
    print(" - has_address_with_district.xlsx")
    print(" - has_address_empty_address.xlsx")

if __name__ == "__main__":
    main()
