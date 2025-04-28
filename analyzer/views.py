import os
import pandas as pd
import numpy as np
from django.shortcuts import render
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from django.conf import settings
import logging
from .models import SalesData
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'uploaded_files')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Set the style for all plots
plt.style.use('default')
sns.set_theme(style="whitegrid", font_scale=1.2)

def format_currency(x, p):
    return f'₹{x:,.0f}'

def create_matplotlib_graph(data, title, xlabel, ylabel, forecast=None, forecast_dates=None, lr_prediction=None):
    plt.figure(figsize=(14, 6))
    plt.plot(data, label='Original Sales', color='blue', marker='o')
    
    if forecast is not None and forecast_dates is not None:
        plt.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
    
    if lr_prediction is not None:
        last_date = data.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        plt.scatter(next_date, lr_prediction, color='green', s=100, label='Next Day Prediction')
        plt.annotate(
            f'Prediction: {lr_prediction:.2f}',
            xy=(next_date, lr_prediction),
            xytext=(30, 30), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', color='black')
        )
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    
    # Convert to base64 string
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return base64.b64encode(image_png).decode('utf-8')

def create_detailed_charts(df):
    # Set higher DPI and font sizes for better clarity
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.autolayout'] = True
    
    # Enhanced axis styling
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['grid.linewidth'] = 1.5
    
    dashboard_charts = {}
    
    # Common function to enhance axes
    def enhance_axes(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=10)
        ax.grid(True, which='major', linestyle='--', linewidth=1.5, alpha=0.3)
        ax.set_axisbelow(True)
    
    # 1. Daily Sales Bar Chart with Trend Line
    plt.figure(figsize=(24, 12))
    daily_sales = df.groupby('Day')['Sales'].sum()
    
    ax = plt.gca()
    enhance_axes(ax)
    
    # Create bar plot with custom style
    bars = ax.bar(range(len(daily_sales)), daily_sales.values, 
                 color=sns.color_palette("husl", 8)[0],
                 alpha=0.8, width=0.7)
    
    # Add trend line with increased visibility
    z = np.polyfit(range(len(daily_sales)), daily_sales.values, 1)
    p = np.poly1d(z)
    plt.plot(range(len(daily_sales)), p(range(len(daily_sales))),
             "r--", alpha=0.9, linewidth=4, label='Trend Line')
    
    plt.title('Daily Sales Analysis with Trend', fontsize=24, pad=20, weight='bold')
    plt.xlabel('Date', fontsize=20, labelpad=15, weight='bold')
    plt.ylabel('Sales', fontsize=20, labelpad=15, weight='bold')
    
    # Format x-axis with more space and larger text
    plt.xticks(range(len(daily_sales)), 
               [d.strftime('%Y-%m-%d') for d in daily_sales.index],
               rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    
    # Add value labels with better visibility
    max_sales = daily_sales.max()
    for i, v in enumerate(daily_sales):
        ax.text(i, v + (max_sales * 0.03), f'{v:,.0f}',
                ha='center', va='bottom', fontsize=16, weight='bold')
    
    plt.legend(loc='upper right', fontsize=16, frameon=True, 
              facecolor='white', edgecolor='black')
    
    plt.subplots_adjust(bottom=0.2, left=0.1)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=400, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    dashboard_charts['daily_sales'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    # 2. Monthly Sales Bar Chart with Revenue
    plt.figure(figsize=(24, 12))
    monthly_data = df.groupby(df['Day'].dt.to_period('M')).agg({
        'Sales': 'sum',
        'Price': lambda x: (x * df.loc[x.index, 'Sales']).sum()
    }).reset_index()
    
    ax = plt.gca()
    enhance_axes(ax)
    
    bars = ax.bar(range(len(monthly_data)), monthly_data['Sales'],
                  color=sns.color_palette("husl", 8)[1],
                  alpha=0.8, width=0.7)
    
    ax2 = ax.twinx()
    ax2.spines['right'].set_linewidth(2)
    ax2.tick_params(axis='y', which='major', labelsize=16, width=2, length=10)
    
    ax2.plot(range(len(monthly_data)), monthly_data['Price'],
             color='red', linewidth=4, marker='o', markersize=15, label='Revenue')
    
    plt.title('Monthly Sales and Revenue Trends', fontsize=24, pad=20, weight='bold')
    ax.set_xlabel('Month', fontsize=20, labelpad=15, weight='bold')
    ax.set_ylabel('Sales', fontsize=20, labelpad=15, weight='bold')
    ax2.set_ylabel('Revenue (₹)', fontsize=20, labelpad=15, weight='bold', color='red')
    
    plt.xticks(range(len(monthly_data)), 
               [str(m) for m in monthly_data['Day']],
               rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16, colors='red')
    
    for i, (sales, revenue) in enumerate(zip(monthly_data['Sales'], monthly_data['Price'])):
        ax.text(i, sales + (monthly_data['Sales'].max() * 0.03), f'{sales:,.0f}',
                ha='center', va='bottom', fontsize=16, weight='bold')
        ax2.text(i, revenue, f'₹{revenue:,.0f}',
                 ha='center', va='bottom', fontsize=16, weight='bold', color='red')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, ['Sales', 'Revenue'], 
              loc='upper right', fontsize=16, frameon=True, 
              facecolor='white', edgecolor='black')
    
    plt.subplots_adjust(bottom=0.2, left=0.1)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=400, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    dashboard_charts['monthly_sales'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    # 3. Day-wise Product Sales (Enhanced Line Chart)
    plt.figure(figsize=(24, 12))
    day_product_sales = df.pivot_table(
        index='Day',
        columns='Product',
        values='Sales',
        aggfunc='sum',
        fill_value=0
    )
    
    ax = plt.gca()
    enhance_axes(ax)
    
    for product in day_product_sales.columns:
        plt.plot(range(len(day_product_sales)), 
                day_product_sales[product], 
                label=product,
                marker='o',
                markersize=10,
                linewidth=3.5)
    
    plt.title('Daily Product-wise Sales Trends', fontsize=24, pad=20, weight='bold')
    plt.xlabel('Date', fontsize=20, labelpad=15, weight='bold')
    plt.ylabel('Sales', fontsize=20, labelpad=15, weight='bold')
    
    plt.xticks(range(len(day_product_sales)), 
               [d.strftime('%Y-%m-%d') for d in day_product_sales.index],
               rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.legend(title='Products', bbox_to_anchor=(1.05, 1), 
              loc='upper left', borderaxespad=0.,
              fontsize=16, title_fontsize=18,
              frameon=True, facecolor='white', edgecolor='black')
    
    plt.subplots_adjust(right=0.85, bottom=0.2, left=0.1)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=400, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    dashboard_charts['day_product_sales'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    # 4. Top Products Pie Chart with Revenue Information
    plt.figure(figsize=(24, 14))
    product_analysis = df.groupby('Product').agg({
        'Sales': 'sum',
        'Price': lambda x: (x * df.loc[x.index, 'Sales']).sum()
    }).sort_values('Sales', ascending=False)

    total_sales = product_analysis['Sales'].sum()
    product_analysis['Percentage'] = (product_analysis['Sales'] / total_sales) * 100

    patches, texts, autotexts = plt.pie(product_analysis['Sales'],
            labels=[f'{idx}\n{int(sales):,} sales\nRevenue: ₹{int(rev):,}' 
                   for idx, (sales, rev) in product_analysis[['Sales', 'Price']].iterrows()],
            autopct=lambda pct: f'{pct:.1f}%',
            startangle=90,
            counterclock=False,
            explode=[0.08] * len(product_analysis),
            colors=sns.color_palette("husl", len(product_analysis)))

    plt.setp(autotexts, size=18, weight='bold')
    plt.setp(texts, size=16, weight='bold')
    
    plt.title('Product Sales Distribution with Revenue', fontsize=24, pad=20, weight='bold')
    plt.axis('equal')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=400, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    dashboard_charts['top_products'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    # 5. Sales by Day of Week
    plt.figure(figsize=(24, 12))
    day_stats = df.groupby('Day_of_week').agg({
        'Sales': ['sum', 'mean', 'count']
    })['Sales']
    
    ax1 = plt.gca()
    enhance_axes(ax1)
    
    bars = ax1.bar(range(len(day_stats)), day_stats['sum'],
                   color=sns.color_palette("husl", 8)[4],
                   alpha=0.8, width=0.7)
    
    ax2 = ax1.twinx()
    ax2.spines['right'].set_linewidth(2)
    ax2.tick_params(axis='y', which='major', labelsize=16, width=2, length=10)
    
    line = ax2.plot(range(len(day_stats)), day_stats['mean'],
                    'r-o', linewidth=4, markersize=15, label='Average Sales')
    
    plt.title('Sales Analysis by Day of Week', fontsize=24, pad=20, weight='bold')
    ax1.set_xlabel('Day of Week', fontsize=20, labelpad=15, weight='bold')
    ax1.set_ylabel('Total Sales', fontsize=20, labelpad=15, weight='bold')
    ax2.set_ylabel('Average Sales', fontsize=20, labelpad=15, weight='bold', color='red')
    
    max_sales = day_stats['sum'].max()
    for i, (total, avg, count) in enumerate(zip(day_stats['sum'],
                                              day_stats['mean'],
                                              day_stats['count'])):
        ax1.text(i, total + (max_sales * 0.03), f'{total:,.0f}\n({count} days)',
                 ha='center', va='bottom', fontsize=16, weight='bold')
        ax2.text(i, avg, f'Avg: {avg:.0f}',
                 ha='center', va='bottom', color='red', fontsize=16, weight='bold')
    
    plt.xticks(range(len(day_stats)), day_stats.index, 
               rotation=45, fontsize=16, weight='bold')
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16, colors='red')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, ['Total Sales', 'Average Sales'],
               loc='upper right', fontsize=16, frameon=True,
               facecolor='white', edgecolor='black')
    
    plt.subplots_adjust(bottom=0.2, left=0.1)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=400, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    dashboard_charts['day_of_week'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    
    return dashboard_charts

def ml_analysis(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('document'):
        file = request.FILES['document']
        filepath = os.path.join(UPLOAD_DIR, file.name)
        
        try:
            # Save the uploaded file
            with open(filepath, 'wb+') as dest:
                for chunk in file.chunks():
                    dest.write(chunk)
            
            logger.debug(f"File saved to: {filepath}")
            
            # Read the CSV file
            df = pd.read_csv(filepath)
            logger.debug(f"CSV columns: {df.columns.tolist()}")
            
            # Check if required columns exist
            if 'Day' not in df.columns or 'Sales' not in df.columns or 'Product' not in df.columns or 'Price' not in df.columns:
                raise ValueError("Missing required columns (Day, Sales, Product, Price)")
            
            # Convert Day column to datetime
            df['Day'] = pd.to_datetime(df['Day'], format='%Y-%m-%d')
            df = df.sort_values(by='Day')
            
            # Truncate existing data and insert new data
            SalesData.objects.all().delete()
            
            # Insert new data
            for _, row in df.iterrows():
                SalesData.objects.create(
                    day=row['Day'],
                    product=row['Product'],
                    sales=row['Sales'],
                    price=row['Price']
                )
            
            # Calculate day numbers for regression
            df['Day_num'] = (df['Day'] - df['Day'].min()).dt.days
            
            # Calculate total sales and revenue
            total_sales = df['Sales'].sum()
            total_revenue = (df['Sales'] * df['Price']).sum()
            
            # Linear Regression Analysis
            X = df[['Day_num']]
            y = df['Sales']
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            next_day_num = df['Day_num'].max() + 1
            lr_prediction = lr_model.predict([[next_day_num]])[0]
            
            # Create dates for the regression line
            regression_dates = pd.date_range(start=df['Day'].min(), periods=len(X) + 1)
            regression_values = lr_model.predict(np.array(range(len(X) + 1)).reshape(-1, 1))
            
            # Create linear regression graph with prediction
            lr_graph = create_matplotlib_graph(
                pd.Series(df['Sales'].values, index=df['Day']),
                'Linear Regression Analysis with Prediction',
                'Date',
                'Sales',
                forecast=regression_values,
                forecast_dates=regression_dates,
                lr_prediction=lr_prediction
            )
            
            # Product Analysis
            product_analysis = []
            for product in df['Product'].unique():
                product_df = df[df['Product'] == product]
                
                # Calculate product metrics
                product_total_sales = product_df['Sales'].sum()
                product_total_revenue = (product_df['Sales'] * product_df['Price']).sum()
                
                # Product trend analysis
                product_X = product_df[['Day_num']]
                product_y = product_df['Sales']
                product_lr = LinearRegression()
                product_lr.fit(product_X, product_y)
                
                # Determine trend
                trend = 'Increasing' if product_lr.coef_[0] > 0 else 'Decreasing'
                trend_strength = round(abs(product_lr.coef_[0]), 2)
                
                # Next day prediction for product
                product_prediction = product_lr.predict([[next_day_num]])[0]
                
                product_analysis.append({
                    'name': product,
                    'total_sales': product_total_sales,
                    'total_revenue': round(product_total_revenue, 2),
                    'trend': trend,
                    'trend_strength': trend_strength,
                    'prediction': round(product_prediction, 2)
                })
            
            # Sort products by total sales
            product_analysis.sort(key=lambda x: x['total_sales'], reverse=True)
            
            # Day of week analysis
            df['Day_of_week'] = df['Day'].dt.day_name()
            day_of_week_sales = df.groupby('Day_of_week')['Sales'].sum().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ]).fillna(0)
            
            # Generate enhanced dashboard charts
            dashboard_charts = create_detailed_charts(df)
            context['dashboard_charts'] = dashboard_charts

            context.update({
                'success': True,
                'filename': file.name,
                'summary': {
                    'total_sales': total_sales,
                    'total_revenue': round(total_revenue, 2),
                    'average_sales': round(df['Sales'].mean(), 2),
                    'max_sales': df['Sales'].max(),
                    'min_sales': df['Sales'].min(),
                    'std_dev': round(df['Sales'].std(), 2),
                    'trend': 'Increasing' if lr_model.coef_[0] > 0 else 'Decreasing',
                    'trend_strength': round(abs(lr_model.coef_[0]), 2)
                },
                'ml_result': {
                    'lr_prediction': round(lr_prediction, 2),
                    'lr_graph': lr_graph,
                    'product_analysis': product_analysis,
                    'day_of_week_data': {
                        'labels': day_of_week_sales.index.tolist(),
                        'values': day_of_week_sales.values.tolist()
                    },
                    'product_category_data': {
                        'labels': [p['name'] for p in product_analysis],
                        'values': [p['total_sales'] for p in product_analysis]
                    }
                }
            })
            
            logger.debug(f"Analysis completed successfully. Context: {context}")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            context['error'] = f"Error processing file: {str(e)}"
    
    return render(request, 'ml_analysis.html', context)

def tableau_dashboard(request):
    return render(request, 'tableau_dashboard.html')
