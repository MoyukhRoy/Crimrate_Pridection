import pandas as pd
import streamlit as st




@st.cache_data 
def load_data():
    data = pd.read_csv('crime.csv')
    return data


data = load_data()
# Create a Streamlit web app
st.title('Crime Data Filter')

# Create input widgets for filtering
crime_type = st.sidebar.selectbox('Select Crime Type:', data['TYPE'].unique())
neighborhood = st.sidebar.selectbox('Select Neighborhood:', data['NEIGHBOURHOOD'].unique())
year = st.sidebar.selectbox('Select Year:', data['YEAR'].unique())

# Filter the data based on user input
filtered_data = data[
    (data['TYPE'] == crime_type) &
    (data['NEIGHBOURHOOD'] == neighborhood) &
    (data['YEAR'] == year)
    ]

# Display the filtered data
st.write(f"Displaying {len(filtered_data)} records")
st.write(filtered_data)

