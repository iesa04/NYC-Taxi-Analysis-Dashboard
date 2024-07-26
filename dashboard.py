import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import pydeck as pdk
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

from timeit import default_timer as timer 

#########################################
st.set_page_config(
    page_title="Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")
#########################################
MONTHS = {
    "January":1,
    "February":2,
    "March":3,
    "April":4,
    "May":5,
    "June":6
}

def read_df(month_number):
    return pd.read_parquet("data/sampled_yellow_trip_data_2023-0{}.parquet".format(month_number),  engine = "fastparquet")

data = pd.DataFrame({
    'x': range(10),
    'y': range(10),
    'category': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
})

#########################################
def format_number(num):
    neg = 0
    if num < 0:
        neg = 1
    if num < 0:
        num = -1 * num
    if num > 1000000:
        if not num % 1000000:
            if neg:
                return f'-{num // 1000000} M'
            else:
                return f'{num // 1000000} M'
        if neg:
            return f'-{round(num / 1000000, 1)} M'
        else:
            return f'{round(num / 1000000, 1)} M'
    if neg:
        return f'-{num // 1000} K'
    else:
        return f'{num // 1000} K'

#########################################

def overview_create_line_chart(df, month_number, line_color):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    df_month = df[(df['tpep_pickup_datetime'] >= '2023-0{}-01'.format(month_number)) & (df['tpep_pickup_datetime'] < '2023-0{}-01'.format(month_number+1))]
    df_month = df_month['tpep_pickup_datetime']

    trips_per_day = df_month.groupby(df_month.dt.date).size().reset_index(name='Trip Count')

    trips_per_day = trips_per_day.rename(columns={'tpep_pickup_datetime': 'Date'})

    fig = px.line(trips_per_day, x='Date', y='Trip Count', color_discrete_sequence=[line_color])
    fig.update_layout(width=650, height = 400)
    st.plotly_chart(fig)

def create_donut_chart_overview(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()

    days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    trip_distribution = df['day_of_week'].value_counts().reindex(days_order)

    day_of_week_to_explode = trip_distribution.idxmax()

    fig = px.pie(trip_distribution, 
                 names=trip_distribution.index, 
                 values=trip_distribution.values,
                 hole=0.6, 
                 labels={'label': 'Day of Week', 'value': 'Trip Count'},
                 )
    
    fig.update_traces(pull=[0.1 if day == day_of_week_to_explode else 0 for day in trip_distribution.index],  
                      textinfo='percent+label',
                      )
    
    fig.update_layout(showlegend=False, title="", width = 410, height = 410)
    
    st.plotly_chart(fig)

def plot_comparison_graphs(month1):
    month_no1 = MONTHS[month1]
 
    df = read_df(month_no1)

    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    # Extract day of the week and hour of the day from the pickup datetime
    df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

    counts_by_hour = df.groupby(['pickup_day_of_week', 'pickup_hour']).size().unstack(fill_value=0)

    counts_by_hour = counts_by_hour.reindex(columns=range(24), fill_value=0)

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    counts_by_hour = counts_by_hour.reindex(index=days_order, fill_value=0)

    plt.figure(figsize=(12, 8))

    for day in days_order:
        plt.plot(counts_by_hour.loc[day].index.astype(str), counts_by_hour.loc[day], label=day)

    plt.xlabel('Hour of Day')
    plt.ylabel('Count of Pickups')
    plt.xticks([i for i in range(24)], [i for i in range(24)])  
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    plt.figure(figsize=(10, 20))
    sns.heatmap(counts_by_hour.T, cmap='YlGnBu', annot = True)  # annot=True to display counts, fmt='d' to format counts as integers
    plt.title('Pickup Counts by Day of Week and Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    st.pyplot(plt)

def plot_ratecode_barchart(selected_month):
    df = read_df(MONTHS[selected_month])

    filtered_df = df[df['RatecodeID'].between(1, 6)]

    ratecode_counts = filtered_df['RatecodeID'].value_counts().reset_index()
    ratecode_counts.columns = ['RatecodeID', 'Count']

    x_axis_labels = ['Standard rate', 'JFK', 'Newark', 'Nassau or Westchester', 'Negotiated fare', 'Group ride']

    fig = px.bar(ratecode_counts, x='RatecodeID', y='Count', 
                color='Count', color_continuous_scale='viridis')
    fig.update_layout(xaxis_title='RatecodeID', yaxis_title='Count', yaxis_type='log')  
    fig.update_xaxes(tickvals=ratecode_counts['RatecodeID'], ticktext=x_axis_labels)
    fig.update_traces(marker_line_color='black', marker_line_width=1, text=ratecode_counts['Count'])  
    fig.update_layout(bargap=0.2, bargroupgap=0.1) 

    st.plotly_chart(fig)

def plot_pie_rates(selected_month):
    df = read_df(MONTHS[selected_month])

    components = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 'airport_fee']

    # Calculate the sum of each component
    fare_breakdown = df[components].sum()

    # Plot interactive pie chart using Plotly Express
    fig = px.pie(names=fare_breakdown.index, values=fare_breakdown.values, 
                labels={'names': 'Components', 'values': 'Amount'}, 
                hole=0.4)

    st.plotly_chart(fig)

def plot_daywise_fare(selected_month):
    df = read_df(MONTHS[selected_month])

    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    filtered_df = df[df['trip_distance'] > 1]

    filtered_df.loc[:, 'pickup_day_of_week'] = filtered_df['tpep_pickup_datetime'].dt.dayofweek

    filtered_df.loc[:, 'cost_per_mile'] = filtered_df['total_amount'] / filtered_df['trip_distance']

    median_cost_per_mile_per_day = filtered_df.groupby('pickup_day_of_week')['cost_per_mile'].median()

    average_duration_per_day = filtered_df.groupby('pickup_day_of_week')['trip_duration'].mean()

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    median_cost_per_mile_per_day.index = days_of_week
    average_duration_per_day.index = days_of_week

    min_duration = average_duration_per_day.min()
    max_duration = average_duration_per_day.max()
    scaled_sizes = 5 + ((average_duration_per_day - min_duration) / (max_duration - min_duration)) * (20 - 5)

    fig = px.scatter(x=median_cost_per_mile_per_day.index, y=median_cost_per_mile_per_day.values,
                    size=scaled_sizes, hover_name=average_duration_per_day.values)

    fig.update_traces(marker=dict(color='orange', symbol='circle'), line=dict(color='teal'), mode='markers+lines')

    hover_text = [f"Median Cost Per Mile: ${y:.2f}<br>Average Trip Duration: {avg_duration:.2f} minutes" 
                for y, avg_duration in zip(median_cost_per_mile_per_day.values, average_duration_per_day.values)]
    fig.update_traces(hovertext=hover_text, hoverinfo='text')

    fig.update_layout(xaxis_title='Day of the Week', yaxis_title='Median Cost Per Mile', showlegend=False)

    st.plotly_chart(fig)

def plot_histograms(selected_month):
    df = read_df(MONTHS[selected_month])

    filtered_df_total_amount = df[(df['total_amount'] > 0) & (df['total_amount'] < 200)]

    fig_total_amount = px.histogram(filtered_df_total_amount, x='total_amount',  nbins = 100)
    fig_total_amount.update_layout(xaxis_title='Total Amount', yaxis_title='Count')
    fig_total_amount.update_traces(marker=dict(color='#2eb853'))

    st.plotly_chart(fig_total_amount)

def show_overview():
    col = st.columns((1, 3.5, 2), gap='medium')

    line_color = "#20a829" 
    with col[0]:
        st.markdown('#### Summary')
        
        months = MONTHS.keys()
        
        selected_month = st.selectbox('Select a Month', months)
    
        if MONTHS[selected_month] == 1:
            df = read_df(MONTHS[selected_month])

            total_trips = format_number(df.shape[0])
            total_miles = format_number(sum(df['trip_distance']))
            total_fare = format_number(sum(df['total_amount']))

            st.metric(label="Total Trips", value=total_trips, delta="")

            st.metric(label="Total Miles", value=total_miles)

            st.metric(label="Total Fare Collected", value="$" + total_fare)
        else:
            df_curr = read_df(MONTHS[selected_month])
            df_prev = read_df(MONTHS[selected_month] - 1)

            total_trips = format_number(df_curr.shape[0])
            total_miles = format_number(sum(df_curr['trip_distance']))
            total_fare = format_number(sum(df_curr['total_amount']))

            total_trips_delta = format_number(df_curr.shape[0] - df_prev.shape[0])
            total_miles_delta = format_number(sum(df_curr['trip_distance']) - sum(df_prev['trip_distance']))
            total_fare_delta = format_number(sum(df_curr['total_amount']) - sum(df_prev['total_amount']))

            if (df_curr.shape[0] - df_prev.shape[0]) < 0:
                line_color = "#f74f4f"

            st.metric(label="Total Trips", value=total_trips, delta=total_trips_delta)

            st.metric(label="Total Miles", value=total_miles, delta=total_miles_delta)

            st.metric(label="Total Fare Collected", value="$" + total_fare, delta=total_fare_delta)
        
    with col[1]:
        st.markdown('#### Trips Per Day in {} 2023'.format(selected_month))
        if MONTHS[selected_month] == 1:
            overview_create_line_chart(df, MONTHS[selected_month], line_color)
        else:
            overview_create_line_chart(df_curr, MONTHS[selected_month], line_color)

        
    with col[2]:
        st.markdown('#### Distribtion of Trips in {} 2023'.format(selected_month))
        if MONTHS[selected_month] == 1:
            create_donut_chart_overview(df)
        else:
            create_donut_chart_overview(df_curr)

def show_comparison():
    col = st.columns((0.5, 2, 2), gap='medium')


    with col[0]:
        months = MONTHS.keys()
        
        selected_comparison_month1 = st.selectbox('Select Month 1', months)
        selected_comparison_month2 = st.selectbox('Select Month 2', months)

    with col[1]:
        st.markdown('#### {} 2023'.format(selected_comparison_month1))
        plot_comparison_graphs(selected_comparison_month1)
        
    with col[2]:
        st.markdown('#### {} 2023'.format(selected_comparison_month2))
        plot_comparison_graphs(selected_comparison_month2)


def show_fare_analysis():
    col = st.columns((0.5, 2, 2), gap='medium')
    

    with col[0]:
        
        months = MONTHS.keys()
        selected_month = st.selectbox('Select Month', months)

    with col[1]:
        st.markdown("# Fare Analysis")
        st.markdown('#### Count Plot of RatecodeID')
        plot_ratecode_barchart(selected_month)

        st.markdown('#### Median Cost Per Mile')
        plot_daywise_fare(selected_month)

    with col[2]:
        for _ in range(5):
            st.markdown("")
        st.markdown("#### Fare Breakdown")
        plot_pie_rates(selected_month)

        st.markdown("#### Distribution of Total Amount")
        plot_histograms(selected_month)

def show_trip_planner():
    col = st.columns((0.5, 2, 2), gap='medium')
    
#########################################
def create_heatmap(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    # Extract day of the week and hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()
    df['hour'] = df['tpep_pickup_datetime'].dt.hour

    # Calculate total passenger count for each combination of day of the week and hour
    heatmap_data = df.groupby(['day_of_week', 'hour'])['passenger_count'].sum().reset_index()

    # Pivot the dataframe to create the heatmap data
    heatmap_data_pivot = heatmap_data.pivot_table(index='day_of_week', columns='hour', values='passenger_count', aggfunc='sum')

    # Reorder the columns to start from Monday
    heatmap_data_pivot = heatmap_data_pivot.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Create the heatmap
    plt.figure(figsize=(20, 6))
    sns.heatmap(heatmap_data_pivot, cmap='viridis', annot=True, fmt='g')
    plt.title('Heatmap of Total Passenger Count by Day of Week and Hour')
    plt.xlabel('Hour')
    plt.ylabel('Day of Week')
    st.pyplot(plt)
##########################################################
def categorize_time_of_day(hour):
    if 0 <= hour < 6:
        return 'Midnight'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Noon'
    else:
        return 'Evening'

def create_donut_chart_time_of_day(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['time_of_day'] = df['tpep_pickup_datetime'].dt.hour.apply(categorize_time_of_day)

    time_of_day_distribution = df['time_of_day'].value_counts()

    fig = px.pie(time_of_day_distribution, 
                 names=time_of_day_distribution.index, 
                 values=time_of_day_distribution.values,
                 hole=0.6, 
                 labels={'label': 'Time of Day', 'value': 'Trip Count'},
                 title="Trips by Time of Day",
                 template="plotly_white",
                 color_discrete_sequence=px.colors.sequential.Viridis)
    
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(showlegend=False, title_x=0.5)
    
    return fig

def show_zones_map():
    # Load the taxi zone dataset
    taxi_zones = gpd.read_file(r"D:\Iesa\MIT\Sem 4\NYC Taxi\taxi_zones\taxi_zones.shp")

    # Calculate centroids
    taxi_zones['centroid'] = taxi_zones.geometry.centroid
    taxi_zones['lat'] = taxi_zones['centroid'].apply(lambda p: p.y)
    taxi_zones['lon'] = taxi_zones['centroid'].apply(lambda p: p.x)

    # Prepare a mapping DataFrame
    location_mapping = taxi_zones[['LocationID', 'lat', 'lon']]

    # Plot the taxi zones on a map
    st.markdown('# NYC Taxi Zones')
    
    col1, col2 = st.columns([1.2, 1])  # Make the first column wider

    with col1:
        for _ in range(2):
            st.markdown("")
        selected_month = st.selectbox("Select a month for zones map", list(MONTHS.keys()), key="zones_month")
        selected_zone_display = st.selectbox("Select a zone", taxi_zones['zone'])
        selected_zone_id = taxi_zones[taxi_zones['zone'] == selected_zone_display]['LocationID'].values[0]
        selected_zone_data = taxi_zones[taxi_zones['LocationID'] == selected_zone_id]
        
        # Load the taxi dataset
        df = read_df(MONTHS[selected_month])

        # Merge pickup coordinates
        df = df.merge(location_mapping, left_on='PULocationID', right_on='LocationID', suffixes=('', '_pickup'))
        df.rename(columns={'lat': 'pickup_latitude', 'lon': 'pickup_longitude'}, inplace=True)

        # Merge dropoff coordinates
        df = df.merge(location_mapping, left_on='DOLocationID', right_on='LocationID', suffixes=('', '_dropoff'))
        df.rename(columns={'lat': 'dropoff_latitude', 'lon': 'dropoff_longitude'}, inplace=True)

        # Filter the dataset based on the selected taxi zone
        df_filtered = df[(df['PULocationID'] == selected_zone_id) | (df['DOLocationID'] == selected_zone_id)]

        # Calculate total pickups and dropoffs
        total_pickups = df_filtered[df_filtered['PULocationID'] == selected_zone_id].shape[0]
        total_dropoffs = df_filtered[df_filtered['DOLocationID'] == selected_zone_id].shape[0]

        # Add total pickups and dropoffs to the DataFrame
        selected_zone_data['Total Pickups'] = total_pickups
        selected_zone_data['Total Dropoffs'] = total_dropoffs

        st.write(f"Selected Zone: {selected_zone_display}")
        st.dataframe(selected_zone_data[['LocationID', 'zone', 'borough', 'Total Pickups', 'Total Dropoffs']], width=2000)  # Adjust width here
        
        for _ in range(15):
            st.markdown("")

        # Hourly Pickup and Dropoff Patterns
        st.markdown('#### Hourly Pickup and Dropoff Patterns')
        df_filtered['pickup_hour'] = pd.to_datetime(df_filtered['tpep_pickup_datetime']).dt.hour
        df_filtered['dropoff_hour'] = pd.to_datetime(df_filtered['tpep_dropoff_datetime']).dt.hour

        hourly_pickup = df_filtered.groupby('pickup_hour').size().reset_index(name='Pickup Count')
        hourly_dropoff = df_filtered.groupby('dropoff_hour').size().reset_index(name='Dropoff Count')

        fig_hourly_pickup = px.line(hourly_pickup, x='pickup_hour', y='Pickup Count', 
                                    title='Hourly Pickup Count',
                                    labels={'pickup_hour': 'Hour of Day', 'Pickup Count': 'Number of Pickups'},
                                    template="plotly_white",
                                    color_discrete_sequence=["#2ca02c"])
        fig_hourly_pickup.update_layout(title_x=0.5)
        st.plotly_chart(fig_hourly_pickup)

        st.markdown('#### Trips by Time of Day')
        fig_time_of_day = create_donut_chart_time_of_day(df_filtered)
        st.plotly_chart(fig_time_of_day)

        st.markdown('#### Trip Duration Distribution')
        df_filtered['trip_duration'] = (pd.to_datetime(df_filtered['tpep_dropoff_datetime']) - pd.to_datetime(df_filtered['tpep_pickup_datetime'])).dt.total_seconds() / 60
        fig3 = px.histogram(df_filtered[df_filtered['trip_duration'] < 200], x='trip_duration', nbins=50, title='Trip Duration Distribution (minutes)', template="plotly_white", color_discrete_sequence=["#d62728"])
        fig3.update_layout(title_x=0.5)
        st.plotly_chart(fig3)

        # Plot 6: Passenger Count Distribution
        st.markdown('#### Passenger Count Distribution')
        fig6 = px.histogram(df_filtered, x='passenger_count', nbins=6, title='Passenger Count Distribution', template="plotly_white", color_discrete_sequence=["#9467bd"])
        fig6.update_layout(title_x=0.5)
        st.plotly_chart(fig6)

        st.markdown('#### Passenger Count by Day of Week and Hour of Day')
        create_heatmap(df_filtered)
     
    with col2:
        
        plt.figure(figsize=(8,8))
        taxi_zones.plot(color='blue', edgecolor='gray', alpha=0.1)
        plt.scatter(selected_zone_data['lon'], selected_zone_data['lat'], color='red', s=10, label='Selected Zone')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('NYC Taxi Zone Map')
        plt.legend()
        st.pyplot(plt)

        fig_hourly_dropoff = px.line(hourly_dropoff, x='dropoff_hour', y='Dropoff Count', 
                                    title='Hourly Dropoff Count',
                                    labels={'dropoff_hour': 'Hour of Day', 'Dropoff Count': 'Number of Dropoffs'},
                                    template="plotly_white",
                                    color_discrete_sequence=["#ff7f0e"])
        fig_hourly_dropoff.update_layout(title_x=0.5)

        for _ in range(3):
            st.markdown("")
        st.plotly_chart(fig_hourly_dropoff)

        st.markdown('#### Total Pick Up and Drop Offs')
        # Plot 1: Total Pickups and Drop-offs per Zone
        total_counts = pd.DataFrame({
            'Type': ['Pickups', 'Drop-offs'],
            'Count': [total_pickups, total_dropoffs]
        })
        fig1 = px.bar(total_counts, x='Type', y='Count', title='Total Pickups and Drop-offs', template="plotly_white", color_discrete_sequence=["#8c564b"])
        fig1.update_layout(title_x=0.5)
        st.plotly_chart(fig1)

        # Plot 2: Trip Distance Distribution
        st.markdown('#### Trip Distance Distribution')
        fig2 = px.histogram(df_filtered, x='trip_distance', nbins=50, title='Trip Distance Distribution', template="plotly_white", color_discrete_sequence=["#e377c2"])
        fig2.update_layout(title_x=0.5)
        st.plotly_chart(fig2)

        # Plot 4: Average Fare by Time of Day
        st.markdown('#### Average Fare by Time of Day')
        df_filtered['pickup_hour'] = pd.to_datetime(df_filtered['tpep_pickup_datetime']).dt.hour
        avg_fare_time_of_day = df_filtered.groupby('pickup_hour')['total_amount'].mean().reset_index()
        fig4 = px.line(avg_fare_time_of_day, x='pickup_hour', y='total_amount', title='Average Fare by Time of Day', template="plotly_white", color_discrete_sequence=["#7f7f7f"])
        fig4.update_layout(title_x=0.5)
        st.plotly_chart(fig4)



def trip_planner():
    contingency_df=pd.read_csv("data/frequency_table.csv")
    contingency_df.drop(columns=['Unnamed: 0'], inplace=True)
    taxi_zones = gpd.read_file("data/taxi_zones.shp")
    taxi_zones['centroid'] = taxi_zones['geometry'].centroid
    location_id_to_zone = taxi_zones.set_index('LocationID')['zone'].to_dict()
    st.markdown("# NYC Taxi Zones with Top Dropoff Locations")

    # Get the zone names for the select box
    zone_names = taxi_zones['zone'].tolist()

    # Function to generate dropdown menu for selecting pickup locations
    def select_pickup_location():
        selected_pickup_location = st.selectbox("Select Pickup Location", zone_names, index = 5)
        return selected_pickup_location

    # Function to display top 5 dropoff locations for the selected pickup location
    def display_top_dropoff_locations(selected_pickup_location):
        st.write(f"Top 5 Dropoff Locations for Pickup Location {selected_pickup_location}:")
        pickup_location_id = taxi_zones[taxi_zones['zone'] == selected_pickup_location]['LocationID'].iloc[0]
        top_dropoff_locations = contingency_df[str(pickup_location_id)].nlargest(5)
        top_dropoff_zones = [location_id_to_zone[int(loc_id)] for loc_id in top_dropoff_locations.index]
        top_dropoff_df = pd.DataFrame({
            'Dropoff Zone': top_dropoff_zones,
            'Frequency': top_dropoff_locations.values
        })

        st.table(top_dropoff_df)

        colors = cm.viridis(np.linspace(0, 1, len(top_dropoff_df))) # Use Viridis colormap
        plt.figure(figsize=(8, 5))
        plt.bar(top_dropoff_df['Dropoff Zone'], top_dropoff_df['Frequency'], color=colors)
        plt.xlabel('Dropoff Zone')
        plt.ylabel('Frequency')
        plt.title('Top 5 Dropoff Zones based on Frequency')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)
        return top_dropoff_df['Dropoff Zone'].tolist() # Return top 5 dropoff locations as a list

    # Create columns for layout
    col1, col2 = st.columns([1.5, 2])

    # Select pickup location
    with col1:
        selected_pickup_location = select_pickup_location()
        print(selected_pickup_location)

        # Display top 5 dropoff locations
        top_dropoff_zones = display_top_dropoff_locations(selected_pickup_location)

    # Get the LocationID of the selected pickup zone
    pickup_location_id = taxi_zones[taxi_zones['zone'] == selected_pickup_location]['LocationID'].iloc[0]
    # Filter taxi zones dataframe to only include the selected pickup and dropoff locations
    selected_zones = taxi_zones[(taxi_zones['LocationID'] == pickup_location_id) | 
                                (taxi_zones['zone'].isin(top_dropoff_zones))]



    # Plot the selected zones on a map using matplotlib
    with col2:
        st.markdown("## Selected Zones Map")
        plt.figure(figsize=(5, 5)) # Adjusted size to be more compact
        taxi_zones.plot(color='orange', edgecolor='gray', alpha=0.1)
        plt.scatter(selected_zones['centroid'].x, selected_zones['centroid'].y, color='blue', s=10, label='Top 5 Zones')
        plt.scatter(selected_zones[selected_zones['zone'] == selected_pickup_location]['centroid'].x, 
                    selected_zones[selected_zones['zone'] == selected_pickup_location]['centroid'].y, 
                    color='red', s=30, label='Selected Zone: ' + selected_pickup_location) # Display selected zone name
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('NYC Taxi Zone Map')
        plt.legend()
        st.pyplot(plt)
######################################################################

with st.sidebar:
    option = option_menu(
        menu_title="Menu",
        options=["Overview", "Zones", "Fare Analysis", "Trip Planner"],
        default_index=0
    )



if option == "Overview":
    start= timer()
    show_overview()
    stop= timer()
    time= stop - start
    print(time)

if option == "Zones":
    show_zones_map()

if option == "Fare Analysis":
    show_fare_analysis()


if option == "Trip Planner":
    frames = [read_df(i) for i in range(1,7)]
    print(len(frames))
    trip_planner()



#########################################

