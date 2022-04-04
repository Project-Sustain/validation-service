import json
import sys


def print_usage():
    print("USAGE: python3 join_shapefile.py <input_path> <output_path>")
    print("Example:\tpython3 join_shapefile.py no_budgets/job_mode_multiprocessing/response.json ~/out.geojson")


def main():

    if len(sys.argv) != 3:
        print_usage()
        exit()

    input_response = sys.argv[1]
    out_path = sys.argv[2]
    counties_shapefile = "counties.geojson"

    with open(counties_shapefile, "r") as f:
        counties = json.load(f)

    with open(input_response, "r") as f:
        responses = json.load(f)

    for worker_response in responses["worker_responses"]:
        for metric_response in worker_response["metrics"]:
            if "ok" in metric_response and metric_response["ok"]:
                gis_join = metric_response["gis_join"]

                # Lookup GISJOIN in counties
                for county in counties["features"]:
                    if county["properties"]["GISJOIN"] == gis_join:
                        county["properties"]["allocation"] = metric_response["allocation"]
                        county["properties"]["duration_sec"] = metric_response["duration_sec"]
                        county["properties"]["variance"] = metric_response["variance"]
                        county["properties"]["loss"] = metric_response["loss"]

    with open(out_path, "w") as f:
        json.dump(counties, f)


if __name__ == "__main__":
    main()

