import subprocess


def main():
    for name in [
        "smooth",
        "sod",
        "lax",
        "leblanc",
        "two_rarefaction",
        "sedov_1959",
        "woodward_colella_1984",
        "shu_osher_1989",
        "shu_osher_1996",
        "linde_roe_1997",
        "titarev_toro_2004",
    ]:
        subprocess.run(["python", f"{name}.py"])


if __name__ == "__main__":
    main()
