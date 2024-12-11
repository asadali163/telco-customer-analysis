# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import math



class Analyzer:

    def __init__(self, df):
        self.df = df
        self.df_churn = self.select_churn_data()

    # Function to make select only churn data
    def select_churn_data(self):
        return self.df[self.df['Churn'] == 'Yes']

    
    # Churn Proportion Checker
    def churn_proportion(self):
        plt.figure(figsize=(5,5), facecolor='lightyellow')
        plt.pie(self.df['Churn'].value_counts(), autopct='%.2f%%', pctdistance = 1.25,startangle=45, textprops={'fontsize': 15},
        colors=['indigo','darkorange'], shadow=True)
        my_circle=plt.Circle( (0,0), 0.6, color='lightyellow')
        p=plt.gcf()
        p.gca().add_artist(my_circle)
        plt.title('Churn Proportion', fontsize=17, fontweight='bold')
        plt.legend(['No', 'Yes'], bbox_to_anchor=(1, 1), fontsize=12)
        plt.show();


    # Function to plot only data that is churn data
    def plot_churn_data(self, cols):
        number_of_columns = 2 
        number_of_rows = (len(cols) + 1) // number_of_columns  
        fig, axes = plt.subplots(number_of_rows, number_of_columns, figsize=(16, 6 * number_of_rows))
        axes = axes.flatten()  

        color_palette = sns.color_palette("pastel")
        for idx, col in enumerate(cols):
            # Plot pie chart
            self.df_churn[col].value_counts().plot.pie(
                autopct='%1.1f%%',
                startangle=90,
                explode=[0.05] * len(self.df_churn[col].unique()),
                ax=axes[idx],
                colors=color_palette[:len(self.df_churn[col].unique())],
                wedgeprops={'edgecolor': 'black'},  # Add black boundary to slices
            )
            axes[idx].set_title(f'{col} Distribution', fontsize=14, weight='bold')
            axes[idx].set_ylabel('')  # Remove default ylabel
            axes[idx].legend(
                self.df_churn[col].unique(), 
                title="Categories", 
                loc='upper right', 
                fontsize=10, 
                title_fontsize=12
            )

        for idx in range(len(cols), len(axes)):
            axes[idx].axis('off')

        fig.tight_layout(pad=3.0)
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Adjust spacing between plots
        plt.show()


    #Make a function to plot categorical data according to target
    def plot_categorical_to_target(self,categorical_values, target):
        number_of_columns = 2
        number_of_rows = math.ceil(len(categorical_values)/2)
        
        fig = plt.figure(figsize = (12, 5*number_of_rows))
        
        for index, column in enumerate(categorical_values, 1):
            ax = fig.add_subplot(number_of_rows,number_of_columns,index)
            ax = sns.countplot(x = column, data = self.df, hue = target, palette="Blues")
            ax.set_title(column)
            # Give some space at the end
            plt.tight_layout()
        return plt.show() 
    

    # Function to plot numeric data
    def plot_numeric_boxplots(self, numeric_cols):
        number_of_columns = 2 
        number_of_rows = (len(numeric_cols) + 1) // number_of_columns  

        fig, axes = plt.subplots(number_of_rows, number_of_columns, figsize=(16, 6 * number_of_rows))
        axes = axes.flatten() 

        for idx, col in enumerate(numeric_cols):
            sns.boxplot(
                data=self.df,
                x='Churn',
                y=col,
                ax=axes[idx],
                palette="pastel",
                width=0.6,
                fliersize=5, 
                linewidth=1.5, 
            )
            axes[idx].set_title(f'{col} Box Plot', fontsize=14, weight='bold')
            axes[idx].set_ylabel(col, fontsize=12)
            axes[idx].set_xlabel('Churn') 

        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        fig.tight_layout(pad=3.0)
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        plt.show()


    # Function for histogram of numerical cols
    def plot_numerical_histogram(self, numeric_cols):
        fig = plt.subplots(figsize=(15, 10))
        for i, col in enumerate(numeric_cols):
            plt.subplot(2, 2, i + 1)
            sns.histplot(data=self.df_churn, x=col, kde=True)
            plt.title(f'{col} distribution of churn customers')
        plt.tight_layout()


    
