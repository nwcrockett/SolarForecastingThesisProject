# conversions to datetime, moving data into a csv, sorting out files
import pandas as pd
# imports data from netcdf to python, removes 2-d and other uneeded data in the import
import xarray as xr
# gets a list of files from the location stored in the file directory
import os

train = [2006, 2007, 2008, 2009, 2010, 2011, 2012]
test = [2015, 2016]
validate = [2013, 2014]


def hourly_data_to_csv(file_path):
    """
    When this runs it is going to be really slow. Keep this in mind. Running only one
    side of these files takes 20 minutes for 4 files.

    Takes the netcdf files from the two sources of hourly data and outputs to csv files.

    Code needs to be adjusted due to directory changes

    :param file_path: path to the netcdf files
    :return: nothing. Outputs 24 csv files
    """
    # sort out the non-hourly data, then between the two data types
    the_files = os.listdir(file_path)
    the_files = pd.Series(the_files)
    needed_files = the_files[the_files.str.startswith("nsaarmbecldradC1")]
    nc_files = needed_files[needed_files.str.endswith("nc")]
    nc_files = nc_files.sort_values()  # Sort files into correct timeseries order
    cdf_files = needed_files[needed_files.str.endswith("cdf")]
    cdf_files = cdf_files.sort_values()  # Sort files into correct timeseries order

    df_2011 = None
    got_other_2011 = False

    year = 2006
    for item in nc_files:
        # Used for naming the file
        print("nc cycles")
        temp_name = item.split(".")

        # Opens up the netcdf data file, drops the below variables
        ds = xr.open_dataset(file_path + "/" + item, drop_variables=[
            "cld_frac", "cld_frac_MPL", "cld_base_source_status",
            "swdif", "lw_net_TOA", "lat",
            "time_bounds", "time_frac", "lon",
            "source_cld_frac", "height", "stdev_sedif"
            "cld_frac_radar", "time", "qc_swdif",
            "swdir", "stdev_swdir", "qc_swdir",
            "swup", "stdev_swup", "qc_swup",
            "lwdn", "stdev_lwdn", "qc_lwdn",
            "lwup", "stdev_lwup", "qc_lwup",
            "sw_net_TOA", "sw_dn_TOA", "totswfluxdn",
            "qc_totswfluxdn", "stdev_totswfluxdn", "alt",
            "qc_cld_frac", "cld_frac_radar"], decode_times=False)

        df = ds.to_dataframe()

        if year == 2011 and got_other_2011:  # fix of a problem not testing at this time. Test later
            df_2011 = df
            got_other_2011 = True
            continue
        if year == 2011 and got_other_2011:
            df = pd.concat([df_2011, df])

        # Add base time (seconds since 1970-1-1 0:00:00 0:00) which in 2006 Jan 1st in this case
        # and the time offset. This will get me a useable timestamp in time_offset that
        # I can make into a datetime
        df["time_offset"] = df["time_offset"] + df["base_time"]
        df["time_offset"] = pd.to_datetime(df["time_offset"], unit='s')

        df = df.drop("base_time", axis=1)
        df = df.set_index("time_offset")
        if year in train:
            df.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/train/"
                      + str(year) + "/PreProcessed_data/"
                      + str(temp_name[0]) + str(temp_name[2])
                      + "nc.csv")
        elif year in validate:
            df.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/validate/"
                      + str(year) + "/PreProcessed_data/"
                      + str(temp_name[0]) + str(temp_name[2])
                      + "nc.csv")
        elif year in test:
            df.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/test/"
                      + str(year) + "/PreProcessed_data/"
                      + str(temp_name[0]) + str(temp_name[2])
                      + "nc.csv")
        year += 1


def nsametc1_to_monthly_files(file_path):
    """
    takes all nsametc1 files. Gets rid of all excess data then puts all other data into
    csv file separated by month

    :param file_path: path to the netcdf files
    :return: returns nothing. Outputs 120 csv files
    """
    the_files = os.listdir(file_path)
    the_files = pd.Series(the_files)
    needed_files = the_files[the_files.str.startswith("nsametC1")]
    needed_files = needed_files.sort_values()
    year = 2006
    month = 1

    while year < 2017:
        dfs = []  # list of DataFrames
        if month < 10:
            month_files = needed_files[needed_files.str.startswith("nsametC1.b1." + str(year) + "0" + str(month))]
        else:
            month_files = needed_files[needed_files.str.startswith("nsametC1.b1." + str(year) + str(month))]

        for item in month_files:
            ds = xr.open_dataset(file_path + "/" + item, drop_variables=[
                    "base_time", "logger_volt", "qc_logger_volt",
                    "logger_temp", "qc_logger_temp", "lat",
                    "lon", "alt", "atmos_pressure",
                    "qc_atmos_pressure", "temp_mean", "qc_temp_mean",
                    "temp_std", "rh_mean", "qc_rh_mean",
                    "rh_std", "vapor_pressure_mean", "qc_vapor_pressure_mean",
                    "vapor_pressure_std", "wspd_arith_mean", "qc_wspd_arith_mean",
                    "wspd_vec_mean", "qc_wspd_vec_mean", "wdir_vec_mean",
                    "qc_wdir_vec_mean", "wdir_vec_std", "pws_pw_code_inst",
                    "qc_pws_pw_code_inst", "pws_pw_code_15min", "qc_pws_pw_code_15min",
                    "pws_pw_code_1hr", "qc_pws_pw_code_1hr", "pws_precip_rate_mean_1min",
                    "qc_pws_precip_rate_mean_1min", "cmh_temp", "qc_cmh_temp",
                    "cmh_dew_point", "qc_cmh_dew_point", "cmh_sat_vapor_pressure",
                    "qc_cmh_sat_vapor_pressure", "cmh_vapor_pressure", "qc_cmh_vapor_pressure",
                    "cmh_rh", "qc_cmh_rh", "trh_err_code",
                    "time_offset"
                                 ])
            df = ds.to_dataframe()
            dfs.append(df)

        monthly_data_df = pd.concat(dfs)
        if month < 10:
            monthly_data_df.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/"
                                   + str(year) + "/nsametC1.b1.0" + str(month) + ".csv")
        else:
            monthly_data_df.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/"
                                   + str(year) + "/nsametC1.b1." + str(month) + ".csv")

        month += 1
        if month == 13:
            month = 1
            year += 1


def nsaradflux1longC1_to_monthly_files(file_path):
    """
    takes all nsaradflux1longC1 files. Gets rid of all excess data then puts all other data into
    csv file separated by month

    :param file_path: path to the netcdf files
    :return: returns nothing. Outputs 120 csv files
    """
    the_files = os.listdir(file_path)
    the_files = pd.Series(the_files)
    needed_files = the_files[the_files.str.startswith("nsaradflux1longC1")]
    needed_files = needed_files.sort_values()
    needed_files = needed_files.reset_index(drop=True)
    year = 2005
    month = 12

    first_ds = xr.open_dataset(file_path + "/" + needed_files[0], drop_variables=[
        "time_bounds", "base_time", "clearsky_downwelling_shortwave",
        "downwelling_longwave", "qc_downwelling_longwave", "clearsky_downwelling_longwave",
        "upwelling_shortwave", "qc_upwelling_shortwave", "clearsky_upwelling_shortwave",
        "upwelling_longwave", "qc_upwelling_longwave", "clearsky_upwelling_longwave",
        "diffuse_downwelling_shortwave", "source_diffuse_downwelling_shortwave", "qc_diffuse_downwelling_shortwave",
        "clearsky_diffuse_downwelling_shortwave", "direct_downwelling_shortwave", "source_direct_downwelling_shortwave",
        "qc_direct_downwelling_shortwave", "clearsky_direct_downwelling_shortwave", "clearsky_status",
        "cloudfraction_longwave", "cloudfraction_shortwave", "cloudfraction_shortwave_status",
        "clearsky_emissivity_longwave", "cosine_zenith", "cloud_transmissivity_shortwave",
        "rh_adjustment_to_clearsky_emissivity_longwave", "lat", "lon",
        "alt", "tau_asymmetry_parameter_status", "tau_temperature_limit_status",
        "ice_cloud_temperature_limit", "time_offset"
        ])
    first_df = first_ds.to_dataframe()
    start_2006_df = first_df["2006"]
    dfs = [start_2006_df]  # list of DataFrames

    while year < 2017:
        if month < 10:
            month_files = needed_files[needed_files.str.startswith("nsaradflux1longC1.c2." + str(year) + "0" + str(month))]
        else:
            month_files = needed_files[needed_files.str.startswith("nsaradflux1longC1.c2." + str(year) + str(month))]

        holder_df = None

        for item in month_files:

            ds = xr.open_dataset(file_path + "/" + item, drop_variables=[
                "time_bounds", "base_time", "clearsky_downwelling_shortwave",
                "downwelling_longwave", "qc_downwelling_longwave", "clearsky_downwelling_longwave",
                "upwelling_shortwave", "qc_upwelling_shortwave", "clearsky_upwelling_shortwave",
                "upwelling_longwave", "qc_upwelling_longwave", "clearsky_upwelling_longwave",
                "diffuse_downwelling_shortwave", "source_diffuse_downwelling_shortwave", "qc_diffuse_downwelling_shortwave",
                "clearsky_diffuse_downwelling_shortwave", "direct_downwelling_shortwave", "source_direct_downwelling_shortwave",
                "qc_direct_downwelling_shortwave", "clearsky_direct_downwelling_shortwave", "clearsky_status",
                "cloudfraction_longwave", "cloudfraction_shortwave", "cloudfraction_shortwave_status",
                "clearsky_emissivity_longwave", "cosine_zenith", "cloud_transmissivity_shortwave",
                "rh_adjustment_to_clearsky_emissivity_longwave", "lat", "lon",
                "alt", "tau_asymmetry_parameter_status", "tau_temperature_limit_status",
                "ice_cloud_temperature_limit", "time_offset"
                ])
            df = ds.to_dataframe()

            if item == month_files.iloc[-1]:
                if month == 12:
                    dfs.append(df[str(year)])
                    holder_df = df[str(year + 1)]
                else:
                    dfs.append(df[str(year) + "-" + str(month)])
                    holder_df = df[str(year) + "-" + str(month + 1)]
            else:
                dfs.append(df)

        monthly_data_df = pd.concat(dfs)
        if year == 2005:
            year += 1
            month = 0
        if month < 10:
            monthly_data_df.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/"
                                   + str(year) + "/nsaradflux1longC1.c2." + str(month) + ".csv")
        else:
            monthly_data_df.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/"
                                   + str(year) + "/nsaradflux1longC1.c2." + str(month) + ".csv")
        dfs = [holder_df]

        month += 1
        if month == 13:
            month = 1
            year += 1


def handle_minute_data(file_path, storage_path):
    """
    Processes the netcdf files and turns them into csv files for each of the two seperate sources.
    Then joins the two files together to a new file.

    :param file_path: path to the netcdf files
    :param storage_path: path to where the csv files made with the netcdf files are stored
    :return: returns nothing. Instead outputs 12 csv files
    """
    nsametc1_to_monthly_files(file_path)
    nsaradflux1longC1_to_monthly_files(file_path)
    files_by_year = os.listdir(storage_path)
    solar_name = "nsaradflux1longC1.c2."
    weather_name = "nsametC1.b1."

    for year in files_by_year:

        for month in range(1, 13):
            files_by_month = os.listdir(storage_path + "/" + year)
            df_solar = pd.read_csv(storage_path + "/" + year + "/" + solar_name + str(month) + ".csv")
            if month < 10:
                df_weather = pd.read_csv(storage_path + "/" + year + "/" + weather_name + "0" + str(month) + ".csv")
            else:
                df_weather = pd.read_csv(storage_path + "/" + year + "/" + weather_name + str(month) + ".csv")

            df_minute_data = df_solar.merge(df_weather)
            df_minute_data.to_csv(storage_path + "/" + year + "/minute_data_" + str(month) + ".csv")


def put_minute_data_into_a_year(file_path):
    files = os.listdir(file_path)
    for year in files:
        csv_files = os.listdir(file_path + "/" + year)  # lists current directory
        csv_files.sort()
        csv_files = pd.Series(csv_files)
        minute_data = csv_files[csv_files.str.startswith("minute")]
        minute_data = minute_data.reindex([0, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3])
        dfs = []
        for item in minute_data:
            df = pd.read_csv(file_path + "/" + year + "/" + item, index_col="time")  # read in csv file of minute data
            df = df.drop(columns=["Unnamed: 0"])
            dfs.append(df)

        yearly_data = pd.concat(dfs, sort=False)
        yearly_data.to_csv(file_path + "/" + year + "/minute_data_total_year.csv")


if __name__ == "__main__":
    ARM_files = "/hdd/ARM files for thesis"
    data_storage = "/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data"
    hourly_data_to_csv(ARM_files)
    # handle_minute_data(ARM_files, data_storage)
    # put_minute_data_into_a_year(data_storage)




