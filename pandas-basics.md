### Pandas:
ITs a Python library used for working with data sets. It has functions for analyzing, cleaning, exploring, and manipulating data

### Pandas Data Types:
Pandas has it's own Data Types with all python primitive Data Types
- Series: 1 Dimensional Data
- Data Frame: Multi Dimensional Data
```py
import pandas as pd

# 2 main datatypes Series (1 Dimensional) and DataFrame (2 Dimensional)
series = pd.Series(["BMW", "Toyota", "Honda"])
colors = pd.Series(["Blue", "Red", "White"]) 
prices = pd.Series(["$1,000", "$2,000", "$3,000"])

# DataFrame = 2 dimensional
car_data = pd.DataFrame({"Car make": series, "Color": colors, "Price": prices})

# convert car_data's price from object to int
car_data["Price"] = car_data["Price"].str.replace(r'[^0-9]', '', regex=True).astype(int)

car_data

#  	Car make 	Color 	Price
# 0 	BMW 	Blue 	1000
# 1 	Toyota 	Red 	2000
# 2 	Honda 	White 	3000
```

### Import/Exporting CSVs, head() and tail():
```py
# Import data
car_sales = pd.read_csv("car-sales.csv")
car_sales.Brand.head() # will take first 5 rows
car_sales.Brand.tail() # will take last 5 rows

# Exporting Data Frame
# index = False to stop forced indexing of the csv rows
car_sales.to_csv("exported-car-sales.csv", index = False)
# car_sales.to_exel("exported-car-sales.xl")
```

### Attributes/Props:
```py
car_sales.dtypes # all the data types of car_sales Data Frame
car_sales.columns # all the column names
car_sales.index # all indexes
```
### Functions/Methods:
```py
car_sales.describe() # give info about a Data Frame like mean(average), min value, max value, total count etc
car_sales.info()
# avarages
car_sales.Price.mean() # average price of the Price column
car_sales["Price"].sum() # total of the price column
len(car_sales) # length of the data frame's row

# .loc & .iloc : Getting data from location id and index location
print(colors.loc[1], colors.loc[2], colors.loc[0])

# Slice Operation
car_sales[:7]
```
### Accessing Column [] vs . :
```py
# dot notation is only possible with column name with no space
car_sales.Brand

# [] notation can receive column name with space
car_sales["Engine Type"]

# accessing multiple column
car_sales[["Engine Type", "Another Column"]]
```

### Filtering || Boolean indexing:
```py
# Showing car sales data where Column "Brand" is only "Toyota"
car_sales[car_sales["Brand"] == "Toyota"]

# Where Mileage is greater than 321
car_sales[car_sales["Mileage"] > 321]
```

### Comparing Columns:
```py
# crosstab
pd.crosstab(car_sales["Brand"], car_sales["Price"]).head()

# groupby
car_sales_sub = car_sales[["Brand","Price"]]
car_sales_sub.groupby(["Brand"]).mean()
```
### Data Frame Visualization:
```py
# using Matplotlib
# %matplotlib inline (inlining matplotlib)
car_sales["Mileage"].plot()

# histogram
car_sales["Mileage"].hist()

# line
car_sales["Price"].head(27).plot(kind="line")
```
### Data Manipulating:
```py
# Copy Data Frame
car_sales_copied = car_sales.copy()

# triming down data sets
car_sales = car_sales.head(21)
# making the brand name lower case
car_sales["Brand"].str.lower()

# Filling Nan/Empty Cells
car_sales[car_sales.isnull().any(axis=1)]

car_sales_fill_task = car_sales.copy()
car_sales_fill_task
# filling a empty cell to average value
car_sales_fill_task["Price"].fillna(car_sales_fill_task["Price"].mean())

# # add average value to Nan field to the existing data frame by inplace=True
car_sales_fill_task["Price"].fillna(car_sales_fill_task["Price"].mean(), inplace=True)
# # this will also work
# # car_sales_fill_task["Price"] = car_sales_fill_task["Price"].fillna(car_sales_fill_task ["Price"].mean())
car_sales_fill_task["Price"]

# Removing Nan/Empty Cell:
car_sales_nan_rm = car_sales.copy()
car_sales_nan_rm.dropna(inplace=True)
car_sales_nan_rm
```

### Deleting and Adding columns to an Exixting DF:
```py
# creating sub data set form a df
car_sales_mod = car_sales[["Brand", "Price", "Mileage"]].head(7).copy()
# add new column to a df suing Series
seat_count = pd.Series([4,5,6,7,8,9,10])
car_sales_mod["Seats"] = seat_count

# add new column to df using python list
fuel_economy = [7, 8, 9, 10, 5, 7, 7]
car_sales_mod["FuelEco"] = fuel_economy
car_sales_mod

# columns form calculated values
car_sales_mod["Total Fule Used"] = car_sales_mod["Mileage"] / car_sales_mod["FuelEco"]
car_sales_mod

# add column using a single value
car_sales_mod["Wheels"] = 4
car_sales_mod

# Dropping a column
car_sales_mod_d = car_sales_mod.drop("Total Fule Used", axis=1).copy()
car_sales_mod_d
```
### Randomizing Data Samples:
```py
# .1 for 10% and 1 for 100%
car_sale_shuffed = car_sales.sample(frac=1)
# reset the randomized data's index also delete the old indexs
car_sale_shuffed.reset_index(drop=True,inplace=True)
car_sale_shuffed
```

### apply(lambda) and column rename:
```py
# changing existing colums value using applay and lambda(annonymous fn)
car_sales_mod["Mileage"] = car_sales_mod["Mileage"].apply(lambda x: x * 1.61)
car_sales_mod

# change Milage to Milage (KM) and convert Mile to KM
car_sales_mod.rename(columns={"Mileage": "Mileage (KM)"}, inplace=True)
car_sales_mod
```