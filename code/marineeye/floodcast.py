import copernicusmarine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import os
from dask.distributed import Client
import warnings
from dask import delayed, compute
from scipy.interpolate import griddata

warnings.filterwarnings('ignore', message='Downloading:')

# Define the delayed function for one day's data fetch + process
@delayed #Decorated with @delayed so that Dask schedules it in parallel.
def fetch_and_process(start_date,end_date,data_id, variable_id,min_long,max_long,min_lat,max_lat):
    year = start_date.year
    month = start_date.month
    day = start_date.day
    start_date_string = f"{year}-{month}-{day}T00:00:00"
    
    year = end_date.year
    month = end_date.month
    day = end_date.day
    end_date_string = f"{year}-{month}-{day}T23:00:00"

    ds = copernicusmarine.read_dataframe(
        dataset_id=data_id,
        variables=[variable_id],
        minimum_longitude=min_long,
        maximum_longitude=max_long,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        start_datetime=start_date_string,
        end_datetime=end_date_string,
    )

    modified_matrix = ds.reset_index()
    modified_matrix = modified_matrix.to_numpy()
    modified_matrix = np.delete(modified_matrix, 0, axis=1)

    return modified_matrix

def hourly_data_2_single_mean(data_matrix):
    chunks = np.array_split(data_matrix, 24, axis=0)
    mean_matrix = np.mean(chunks,axis=0)
    return mean_matrix    

def calc_monthly_mean(year,min_long,max_long,min_lat,max_lat):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    [alt_comb_data_calib, wind_stress_comb_data_calib, air_density_comb_data_calib] = collect_data(start_date, end_date,min_long,max_long,min_lat,max_lat)

    if year%4==0:
        ranges = [(0, 31), (31, 60), (60, 91), (91,121), (121,152), (152,182), (182,213), (213,243), (243,274), (274,304), (304,335), (335,365)] 
        total_days = 366
    else:
        ranges = [(0, 31), (31, 59), (59, 90), (90,120), (120,151), (151,181), (181,212), (212,242), (242,273), (273,303), (303,334), (334,364)]
        total_days = 365

    alt_comb_data_chunks = np.array_split(alt_comb_data_calib,total_days,axis=0)
    wind_stress_comb_data_chunks = np.array_split(wind_stress_comb_data_calib,total_days,axis=0)
    air_density_comb_data_chunks = np.array_split(air_density_comb_data_calib,total_days,axis=0)

    #Calculate means over these ranges
    alt_means = [np.mean(alt_comb_data_chunks[start:end], axis=0) for start, end in ranges]
    wind_stress_means = [np.mean(wind_stress_comb_data_chunks[start:end], axis=0) for start, end in ranges]
    density_means = [np.mean(air_density_comb_data_chunks[start:end], axis=0) for start, end in ranges]

    final_path = os.path.join("data", "floodcast", "Monthly_Averages", "Meta_Data_Monthly_Avg.csv")

    for i in range(len(alt_means)):
        final_path = os.path.join("data", "floodcast", "Monthly_Averages", "Altitude",f"SSH_Monthly_Average_Data_M{i+1}.csv")
        np.savetxt(final_path, alt_means[i], delimiter=",",fmt=('%.4f', '%.4f', '%.5f'))
        final_path = os.path.join("data", "floodcast", "Monthly_Averages", "Northward_Velocity",f"Velo_Monthly_Average_Data_M{i+1}.csv")
        np.savetxt(final_path, wind_stress_means[i], delimiter=",",fmt=('%.4f', '%.4f', '%.5f'))
        final_path = os.path.join("data", "floodcast", "Monthly_Averages", "Air_Density",f"Air_D_Average_Data_M{i+1}.csv")
        np.savetxt(final_path, density_means[i], delimiter=",",fmt=('%.4f', '%.4f', '%.5f'))

def normalize_matrix(data_matrix,monthly_mean_data_matrix):
    #Normalizing the Matrices
    data_min = np.nanmin(data_matrix)
    data_max = np.nanmax(data_matrix)
    mean_data_min = np.nanmin(monthly_mean_data_matrix)
    mean_data_max = np.nanmax(monthly_mean_data_matrix)
    true_min = min(data_min,mean_data_min)
    true_max = max(data_max,mean_data_max)

    norm_data_matrix = (data_matrix - true_min)/(true_max-true_min)
    norm_monthly_data_matrix = (monthly_mean_data_matrix - true_min)/(true_max-true_min)

    return norm_data_matrix,norm_monthly_data_matrix

def AHP_and_Anomaly_process(month, alt_comb_data, wind_stress_comb_data, air_density_comb_data,rel_weight_1,rel_weight_2,rel_weight_3):
    alt_weight_wrt_wind_stress = rel_weight_1
    alt_weight_wrt_air_density = rel_weight_2
    wind_stress_weight_wrt_air_density = rel_weight_3
    
    weight_matrix = [[1,alt_weight_wrt_wind_stress,alt_weight_wrt_air_density],[(1/alt_weight_wrt_wind_stress),1,wind_stress_weight_wrt_air_density],[(1/alt_weight_wrt_air_density),(1/wind_stress_weight_wrt_air_density),1]]
    eigenvalues, eigenvectors = np.linalg.eig(weight_matrix)

    #Getting the index of maximum eigen value
    max_index = np.argmax(eigenvalues)

    # Get the corresponding eigenvector
    principal_eigenvector = eigenvectors[:, max_index]

    alt_weight = principal_eigenvector[0].real/(principal_eigenvector.sum()).real
    wind_stress_weight = principal_eigenvector[1].real/(principal_eigenvector.sum()).real
    air_density_weight = principal_eigenvector[2].real/(principal_eigenvector.sum()).real

    # Consistency check
    λ_max = eigenvalues[max_index].real
    CI = (λ_max - 3) / 2
    RI = 0.58  # for n=3
    CR = CI / RI

    if CR > 0.1:
        CR_comment = f"⚠️ Warning: Judgments may be inconsistent (CR &lt; 0.1)<br>Consistency Ratio (CR): {CR}"
    else:
        CR_comment = f"Judgments for AHP Process are Consistent (CR &lt; 0.1)<br>Consistency Ratio (CR): {CR}"

    final_path = os.path.join("data", "floodcast", "Monthly_Averages", "Altitude",f"SSH_Monthly_Average_Data_M{month}.csv")
    alt_monthly_mean_matrix = np.loadtxt(final_path, delimiter=',')
    final_path = os.path.join("data", "floodcast", "Monthly_Averages", "Northward_Velocity",f"Velo_Monthly_Average_Data_M{month}.csv")    
    wind_stress_monthly_mean_matrix = np.loadtxt(final_path, delimiter=',')
    final_path = os.path.join("data", "floodcast", "Monthly_Averages", "Air_Density",f"Air_D_Average_Data_M{month}.csv")   
    air_density_monthly_mean_matrix = np.loadtxt(final_path, delimiter=',')

    #Normalizing the Matrices
    [alt_norm_data, alt_norm_monthly_mean_matrix] = normalize_matrix(alt_comb_data[:,2],alt_monthly_mean_matrix[:,2])
    [wind_stress_norm_data, wind_stress_norm_monthly_mean_matrix] = normalize_matrix(wind_stress_comb_data[:,2],wind_stress_monthly_mean_matrix[:,2])
    [air_density_norm_data, air_density_norm_monthly_mean_matrix] = normalize_matrix(air_density_comb_data[:,2],air_density_monthly_mean_matrix[:,2])

    #Multiplying the northward velocity by -1, cause coast of Jartaka is north facing and we need south ward velocity
    wind_stress_norm_data = -1*wind_stress_norm_data
    wind_stress_norm_monthly_mean_matrix = -1*wind_stress_norm_monthly_mean_matrix
    
    #Inversing the values of air density, cause low density->low pressure->high flood chances
    air_density_norm_data = 1-air_density_norm_data
    air_density_norm_monthly_mean_matrix = 1-air_density_norm_monthly_mean_matrix

    alt_AHP_val = alt_norm_data*alt_weight
    wind_stress_AHP_val = wind_stress_norm_data*wind_stress_weight 
    air_density_AHP_val = air_density_norm_data*air_density_weight

    monthly_mean_alt_AHP_val = alt_norm_monthly_mean_matrix*alt_weight
    monthly_mean_wind_stress_AHP_val = wind_stress_norm_monthly_mean_matrix*wind_stress_weight 
    monthly_mean_air_density_AHP_val = air_density_norm_monthly_mean_matrix*air_density_weight

    alt_AHP_val = alt_AHP_val.reshape(-1,1) 
    wind_stress_AHP_val = wind_stress_AHP_val.reshape(-1,1) 
    air_density_AHP_val = air_density_AHP_val.reshape(-1,1) 

    monthly_mean_alt_AHP_val = monthly_mean_alt_AHP_val.reshape(-1,1) 
    monthly_mean_wind_stress_AHP_val = monthly_mean_wind_stress_AHP_val.reshape(-1,1) 
    monthly_mean_air_density_AHP_val = monthly_mean_air_density_AHP_val.reshape(-1,1) 

    alt_AHP_val = np.array(alt_AHP_val, dtype=float)
    wind_stress_AHP_val = np.array(wind_stress_AHP_val, dtype=float)
    air_density_AHP_val = np.array(air_density_AHP_val, dtype=float)

    monthly_mean_alt_AHP_val = np.array(monthly_mean_alt_AHP_val, dtype=float)
    monthly_mean_wind_stress_AHP_val = np.array(monthly_mean_wind_stress_AHP_val, dtype=float)
    monthly_mean_air_density_AHP_val = np.array(monthly_mean_air_density_AHP_val, dtype=float)

    mask_1 = np.isnan(alt_AHP_val) | np.isnan(wind_stress_AHP_val) | np.isnan(air_density_AHP_val)
    mask_2 = np.isnan(monthly_mean_alt_AHP_val) | np.isnan(monthly_mean_wind_stress_AHP_val) | np.isnan(monthly_mean_air_density_AHP_val)

    # Sum arrays normally (ignores NaNs)
    AHP_processed_array = np.nansum([alt_AHP_val, wind_stress_AHP_val, air_density_AHP_val], axis=0)
    monthly_mean_AHP_processed_array = np.nansum([monthly_mean_alt_AHP_val, monthly_mean_wind_stress_AHP_val, monthly_mean_air_density_AHP_val], axis=0)

    # Set positions with any NaN to 0
    AHP_processed_array[mask_1] = 0
    monthly_mean_AHP_processed_array[mask_2] = 0
    anomaly_matrix = AHP_processed_array-monthly_mean_AHP_processed_array
    anomaly_matrix[anomaly_matrix < 0] = 0

    #Adding the lat and long columns to the final matrix
    AHP_processed_array = np.concatenate([alt_comb_data[:,0:1], alt_comb_data[:,1:2], AHP_processed_array],axis=1)
    monthly_mean_AHP_processed_array = np.concatenate([alt_comb_data[:,0:1], alt_comb_data[:,1:2], monthly_mean_AHP_processed_array],axis=1)
    anomaly_matrix = np.concatenate([alt_comb_data[:,0:1], alt_comb_data[:,1:2], anomaly_matrix],axis=1)

    return AHP_processed_array, anomaly_matrix, monthly_mean_AHP_processed_array,CR_comment

def plot_heat_map(processed_matrix,anomaly_matrix,monthly_mean_AHP):
    # Flatten
    lat = processed_matrix[:,0].flatten()
    lon = processed_matrix[:,1].flatten()
    val1 = processed_matrix[:,2].flatten()
    val2 = anomaly_matrix[:,2].flatten()
    val3 = monthly_mean_AHP[:,2].flatten()

    # Create grid
    num_grid = 100  # grid resolution
    lon_grid = np.linspace(lon.min(), lon.max(), num_grid)
    lat_grid = np.linspace(lat.min(), lat.max(), num_grid)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Interpolate scattered data onto grid
    val1_grid = griddata((lon, lat), val1, (lon_grid, lat_grid), method='linear')
    val2_grid = griddata((lon, lat), val2, (lon_grid, lat_grid), method='linear')
    val3_grid = griddata((lon, lat), val3, (lon_grid, lat_grid), method='linear')

    # Plot
    figs,axs = plt.subplots(3,1,figsize=(5,6),subplot_kw={'projection': ccrs.PlateCarree()},constrained_layout=True)

    # First sub-plot
    ax=axs[0]
    ax.add_feature(cfeature.COASTLINE)
    white_coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '10m',
        edgecolor='white', facecolor='none'
    )
    ax.add_feature(white_coastline, linestyle=':')
    mesh1 = ax.pcolormesh(lon_grid, lat_grid, val1_grid, cmap='plasma', shading='auto', transform=ccrs.PlateCarree())
    plt.colorbar(mesh1, ax=ax, orientation='vertical',label='Flood Index')
    
    # Add gridlines with lat/lon labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(0.3)   # longitude every 0.3 degrees
    gl.ylocator = mticker.MultipleLocator(0.1)   # latitude every 0.1 degrees

    ax.set_title("Flood Index Map")


    # First sub-plot
    ax=axs[1]
    ax.add_feature(cfeature.COASTLINE)
    white_coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '10m',
        edgecolor='white', facecolor='none'
    )
    ax.add_feature(white_coastline, linestyle=':')
    mesh1 = ax.pcolormesh(lon_grid, lat_grid, val3_grid, cmap='plasma', shading='auto', transform=ccrs.PlateCarree())
    plt.colorbar(mesh1, ax=ax, orientation='vertical',label='Flood Index')

    # Add gridlines with lat/lon labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(0.3)   # longitude every 0.3 degrees
    gl.ylocator = mticker.MultipleLocator(0.1)   # latitude every 0.1 degrees

    ax.set_title("Mean Flood Index Map for the Month")

    # Second sub-plot
    ax=axs[2]
    ax.add_feature(cfeature.COASTLINE)
    white_coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '10m',
        edgecolor='white', facecolor='none'
    )
    ax.add_feature(white_coastline, linestyle=':',edgecolor='white')
    mesh2 = ax.pcolormesh(lon_grid, lat_grid, val2_grid, cmap='plasma', shading='auto', transform=ccrs.PlateCarree())
    plt.colorbar(mesh2, ax=ax, orientation='vertical',label='Flood Index Range')

    # Add gridlines with lat/lon labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(0.3)   # longitude every 0.3 degrees
    gl.ylocator = mticker.MultipleLocator(0.1)   # latitude every 0.1 degrees

    ax.set_title("Anomaly of Current & Mean Flood Index")
    
    # plt.tight_layout()
    final_path = os.path.join("icons", "Flood_Anomaly_Map.png")
    figs.savefig(final_path, dpi=100,bbox_inches='tight')
    plt.close()


def collect_data(start_date,end_date,min_long,max_long,min_lat,max_lat):
    copernicusmarine.login(username='', password='',check_credentials_valid=True, force_overwrite= False) #Authentication process to access data from copernicus
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Start Dask distributed client (local cluster)
    client = Client()
    print(f"Dask client started with dashboard link: {client.dashboard_link}")

    # Create delayed tasks for each day
    task1 = [fetch_and_process(start_date,end_date,"cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D","adt",min_long,max_long,min_lat,max_lat)]
    task2 = [fetch_and_process(start_date,end_date,"cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H","northward_stress",min_long,max_long,min_lat,max_lat)]
    task3 = [fetch_and_process(start_date,end_date,"cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H","air_density",min_long,max_long,min_lat,max_lat)]

    # Compute all tasks in parallel
    all_results = compute(*task1, *task2, *task3)
    result1 = all_results[:len(task1)]
    result2 = all_results[len(task1):(len(task1)+len(task2))]
    result3 = all_results[(len(task1)+len(task2)):]

    # Stack results into 3D NumPy array: shape = (rows, cols, i_max)
    alt_comb_data = np.concatenate(result1, axis=0)
    wind_stress_comb_data = np.concatenate(result2, axis=0)
    air_density_comb_data = np.concatenate(result3, axis=0)

    day_count = (end_date - start_date).days + 1
    split_size = air_density_comb_data.shape[0]/(24*(alt_comb_data.shape[0]/day_count))
    chunks = np.array_split(air_density_comb_data, split_size, axis=0)
    task4 = [hourly_data_2_single_mean(chunk) for chunk in chunks];
    result4 = compute(*task4)
    air_density_comb_data = np.concatenate(result4, axis=0)

    chunks = np.array_split(wind_stress_comb_data, split_size, axis=0)
    task5 = [hourly_data_2_single_mean(chunk) for chunk in chunks];
    result5 = compute(*task5)
    wind_stress_comb_data = np.concatenate(result5, axis=0)

    temp_matrix = pd.DataFrame(alt_comb_data)
    final_path = os.path.join("data", "floodcast", "Altitude_Prev_Test_File","copernicus_adt_data.csv")
    temp_matrix.to_csv(final_path, index=False)
    
    temp_matrix = pd.DataFrame(wind_stress_comb_data)
    final_path = os.path.join("data", "floodcast", "Northward_Velocity_Prev_Test_File","copernicus_wind_stress_data.csv")  
    temp_matrix.to_csv(final_path, index=False)

    temp_matrix = pd.DataFrame(air_density_comb_data)
    final_path = os.path.join("data", "floodcast", "Air_Density_Prev_Test_File","copernicus_air_density_data.csv") 
    temp_matrix.to_csv(final_path, index=False)
    
    # Close the client when done
    client.close()

    return alt_comb_data, wind_stress_comb_data, air_density_comb_data