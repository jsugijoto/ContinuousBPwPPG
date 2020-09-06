#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import heartpy as hp


# In[6]:


# Modular creation to analyze any signal, given the data itself in CSV format and the 
# frequency that its taken at

class Filter:
    
    # Constructor that needs the data itself, opened with pd.open_csv()
    def __init__(self, data, fs): 
        self.data = data.value
        self.df = data
        self.fs = fs
        self.filtered_ppg = hp.filter_signal(self.data, cutoff = [0.5, 10], filtertype = 'bandpass', 
                                       sample_rate = self.fs, order = 6, return_top = False)
        
    # Displays the datetime and values of the signal only    
    def header(self): 
        df = pd.DataFrame(self.df,columns = ['datetime.$date', 'value'])
        df['datetime.$date'] = pd.to_datetime(df['datetime.$date'], unit = "ms")
        return df
    
    # Displays original PPG signals from data (UNFILTERED PPG)
    def original(self):
        plt.title('Original PPG')
        plt.plot(self.data)
    
    # Filters the PPG signal for between 48 BPM and 180 BPM (0.8 Hz and 2.5 Hz) using a bandpass filter of order 3
    def display(self):
        plt.plot(self.filtered_ppg)
        plt.title('Filtered PPG')
        plt.show()
        
    # Displays the extracted heart rate from the PPG signal using modules from the HeartPY library    
    def extract_HR(self): 
        wd, m = hp.process(self.filtered_ppg, sample_rate=self.fs,
                   high_precision = True)
        hp.plotter(wd, m)
    
    # Manual zoom - x1 is lower x bound, x2 is upper x bound, y1 is lower y bound, y2 is upper y bound FOR HR GRAPH
    def manual_zoomHR(self, x1, x2, y1, y2): 
        plt.xlim(x1,x2)
        plt.ylim(y1,y2)
        self.extract_HR()
        
     # Displays original CSV file that was loaded with pd.open_csv()    
    def display_csv(self):
        return self.df
    
    def manual_zoom_original(self, x1, x2, y1, y2): 
        plt.xlim(x1,x2)
        plt.ylim(y1,y2)
        self.original()    
        
    def manual_zoom_filtered(self, x1, x2, y1, y2): 
        plt.xlim(x1,x2)
        plt.ylim(y1,y2)
        self.display() 

    def out(self):
    	return self.filtered_ppg


# In[9]:


# Testing against PPG data that Milad took with a heart rate at the same time
#test1 = pd.read_csv(r"C:\Users\Jonathan\Desktop\Code\Summer2020\preprocessing\Verify\data-ppg.csv")
#testing1 = Filter(test1, 26)
# testing1.header()
# testing1.display_csv()
# testing1.extract_HR()
# testing1.original()
#testing1.manual_zoom_filtered(12800,13000, -15000,15000)

# Working as of August 4, 2020


# In[4]:


# Testing with original PPG data that was given to me from Milad
#test2 = pd.read_csv(r"C:\Users\Jonathan\Desktop\Code\Summer2020\preprocessing\ppg-milad[2538].csv")

#testing2 = Filter(test2, 25)
# testing2.header()
# testing2.display_csv()
#testing2.extract_HR()
# testing2.original()
# testing2.display()

# Working as of August 4, 2020


# In[ ]:




