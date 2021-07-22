from PySAM.PySSC import *
import pandas as pd
import numpy as np
import requests,io
from scipy.stats import norm
from scipy import linalg
from math import sqrt,exp,log,cos,pi
from scipy.linalg import matmul_toeplitz



class SolarSite(object):
    """ Pull NSRDB site by sending a request.
    Parameters
    ----------
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    
    """
    def __init__(self, lat,lon):
        self.lat = lat
        self.lon = lon

    def get_nsrdb_data(self,year,leap_year,interval,utc):
        """ Pull NSRDB site by sending a request.
        Parameters
        ----------
        year : int
            Choose year of data. May take any value in the interval [1998,2019]
        leap_year : bool
            Set leap year to true or false. True will return leap day data if present, false will not.
        interval: string
            Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
        utc : bool
            Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
            NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
            local time zone.

        """
        
        api_key = 'KgmyZTqyzgQOOjuWoMxPgMEVVMAG0kMV521gJPVv' # NSRDB api key
        attributes = 'ghi,clearsky_ghi,dhi,clearsky_dhi,dni,clearsky_dni,wind_speed,air_temperature,cloud_type,fill_flag,wind_direction'
        your_name = 'pySODA' # Your full name, use '+' instead of spaces.
        reason_for_use = 'Distributed+PV+Generation' # Your reason for using the NSRDB.
        your_affiliation = 'LBNL-ASU' # Your affiliation
        your_email = 'ilosadac@asu.edu' # Your email address
        mailing_list = 'false'
        
        # Declare url string
        url = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'\
        .format(year=year, lat=self.lat, lon=self.lon, leap=str(leap_year).lower(), interval=interval,\
                utc=str(utc).lower(), name=your_name, email=your_email, mailing_list=mailing_list, \
                affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
        
        # Return just the first 2 lines to get metadata:
        r=requests.get(url)
        if r.status_code==400:
            raise NameError(r.json()["errors"][0])
        meta = pd.read_csv(io.StringIO(r.content.decode('utf-8')),nrows=1).T
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')),skiprows=2)
        idx = pd.date_range(start='1/1/{yr}'.format(yr=year), freq=interval+'Min', end='12/31/{yr} 23:59:00'.format(yr=year))
        if leap_year==False:
            idx = idx[(idx.day != 29) | (idx.month != 2)]
        df=df.set_index(idx)

        self.resource_data = df
        self.meta_resource_data = meta.T.to_dict('r')[0]
        
        return df
    
    def generate_solar_power_from_nsrdb(self,clearsky,capacity,DC_AC_ratio,tilt,azimuth,inv_eff,losses,array_type,year = None,leap_year = None,interval = None,utc = None):
        """ Generate PV power time series.
        Parameters
        ----------
        clearsky : bool
            True returns clearsky power, false returns "simulated" output
        capacity : float
            System capacity in MW
        DC_AC_ratio : float
            DC/AC ratio (or power ratio). See https://sam.nrel.gov/sites/default/files/content/virtual_conf_july_2013/07-sam-virtual-conference-2013-woodcock.pdf
        tilt : float
           Tilt of system in degrees
        azimuth : float
            Azimuth angle (in degrees) from north (0 degrees)
        inv_eff : float
            Inverter efficiency (in %)
        losses : float
            Total system losses (in %)
        array_type : int
            # Specify PV configuration (0=Fixed, 1=Fixed Roof, 2=1 Axis Tracker, 3=Backtracted, 4=2 Axis Tracker)
        year : int
            Year of data. May take any value in the interval [1998,2018]
        leap_year : bool
            Leap year to true or false. True will return leap day data if present, false will not.
        interval: string
            Time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
        utc : bool
            Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
            NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
            local time zone.

        """
        if not hasattr(self,"resource_data"):
            args = [arg==None for arg in [year,leap_year,interval,utc]]
            if any(args):
                raise NameError("Missing input: year,leap_year,interval,utc")
            else:
                self.year = year
                self.leap_year = leap_year
                self.interval = interval
                self.utc = utc
                self.get_nsrdb_data(self)

        if clearsky==True:
            clearsky_str="Clearsky "
        else:
            clearsky_str=""

        ssc = PySSC()

        # Resource inputs for SAM model:
        wfd = ssc.data_create()
        ssc.data_set_number(wfd, 'lat'.encode('utf-8'), self.lat)
        ssc.data_set_number(wfd, 'lon'.encode('utf-8'), self.lon)
        ssc.data_set_number(wfd, 'tz'.encode('utf-8'), self.meta_resource_data["Time Zone"])
        ssc.data_set_number(wfd, 'elev'.encode('utf-8'), self.meta_resource_data["Elevation"])
        ssc.data_set_array(wfd, 'year'.encode('utf-8'), self.resource_data.Year)
        ssc.data_set_array(wfd, 'month'.encode('utf-8'), self.resource_data.Month)
        ssc.data_set_array(wfd, 'day'.encode('utf-8'), self.resource_data.Day)
        ssc.data_set_array(wfd, 'hour'.encode('utf-8'), self.resource_data.Hour)
        ssc.data_set_array(wfd, 'minute'.encode('utf-8'), self.resource_data.Minute)
        ssc.data_set_array(wfd, 'dn'.encode('utf-8'), self.resource_data["{}DNI".format(clearsky_str)])
        ssc.data_set_array(wfd, 'df'.encode('utf-8'), self.resource_data["{}DHI".format(clearsky_str)])
        ssc.data_set_array(wfd, 'wspd'.encode('utf-8'), self.resource_data['Wind Speed'])
        ssc.data_set_array(wfd, 'tdry'.encode('utf-8'), self.resource_data.Temperature)

        # Create SAM compliant object  
        dat = ssc.data_create()
        ssc.data_set_table(dat, 'solar_resource_data'.encode('utf-8'), wfd)
        ssc.data_free(wfd)

        # Specify the system Configuration

        ssc.data_set_number(dat, 'system_capacity'.encode('utf-8'), capacity)
        ssc.data_set_number(dat, 'dc_ac_ratio'.encode('utf-8'), DC_AC_ratio)
        ssc.data_set_number(dat, 'tilt'.encode('utf-8'), tilt)
        ssc.data_set_number(dat, 'azimuth'.encode('utf-8'), azimuth)
        ssc.data_set_number(dat, 'inv_eff'.encode('utf-8'), inv_eff)
        ssc.data_set_number(dat, 'losses'.encode('utf-8'), losses)
        ssc.data_set_number(dat, 'array_type'.encode('utf-8'), array_type)
        ssc.data_set_number(dat, 'gcr'.encode('utf-8'), 0.4) # Set ground coverage ratio
        ssc.data_set_number(dat, 'adjust:constant'.encode('utf-8'), 0) # Set constant loss adjustment

        # execute and put generation results back into dataframe
        mod = ssc.module_create('pvwattsv5'.encode('utf-8'))
        ssc.module_exec(mod, dat)
        df=pd.DataFrame()
        df['generation'] = np.array(ssc.data_get_array(dat, 'gen'.encode('utf-8')))
        df.index = self.resource_data.index

        # free the memory
        ssc.data_free(dat)
        ssc.module_free(mod)

        self.cloud_type = self.resource_data['Cloud Type']
        self.solar_power_from_nsrdb = df
        self.capacity = capacity
        
        return df
    
    def generate_high_resolution_power_data(self, resolution, date):
        """ Generate PV power time series.
        Parameters
        ----------
        resolution : string
            Resolution of time series. Recommended values are "1S","30S","1min","5min". For more examples, see Pandas DateOffset objects
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        date : string
            Date in YYYY-MM-DD format. For example "2015-07-14"
            
        """
        date_index2 = pd.date_range(start='2015-01-10', end = '2015-01-10' + ' 23:59:59', freq='1S')
        ts = self.solar_power_from_nsrdb[date].resample("1S").interpolate(method="linear")
        ts = ts.reindex(date_index2).fillna(0)

        ts *= (7.5/self.capacity)

        ct = self.cloud_type[date].resample("1S").pad()
        ct = ct.reindex(date_index2).fillna(0)
        ct = ct.astype(int)


        σ = 0.0003447

        λm = np.array([999999999, 999999999  , 3.2889645, 3.9044665, 3.2509495, 0, 4.1906035, 3.097432 , 4.088177,3.9044665,999999999,3.2889645,3.2889645])
        λw = np.array([5.977229, 5.804869, 6.503102, 6.068099, 5.879129, 0, 4.834679, 5.153073, 6.661633,6.068099,5.977229,6.503102,6.503102])

        pm = np.array([0.001250, 0.002803, 0.009683, 0.005502, 0.018888, 0, 0.000432, 0.007383, 0.003600,0.005502,0.001250,0.009683,0.009683])
        pw = np.array([0.001941, 0.008969, 0.003452, 0.002801, 0.004097, 0, 0.001111, 0.004242, 0.008000,0.002801,0.001941,0.003452,0.003452])

        df = ts[ts.values>0]
        df.insert(1, "CloudType", ct[df.index].values)

        M_hat = 600
        N = len(df)
        # N = 86400
        hm = np.array([exp(-t**2/2)*cos(5*t) for t in np.linspace(-4,4,M_hat)])
        hw = np.array([0.54-0.46*cos(2*pi*t/(M_hat-1)) for t in range(0,M_hat)])

        padding1 = np.zeros(N - M_hat, hm.dtype)
        padding2 = np.zeros(N - M_hat - 1, hm.dtype)

        first_col1 = np.r_[hm, padding1]
        first_row1 = np.r_[hm[0], padding2]

        first_col2 = np.r_[hw, padding1]
        first_row2 = np.r_[hw[0], padding2]


        zw = []
        zm = []
        η = np.zeros(N)
        for i in range(0,N-M_hat):
            if df["CloudType"].values[i]<2:
                zm.append(0)
                zw.append(0)
            else:
                zm.append(np.random.exponential(1/λm[df["CloudType"].values[i]]))
                zw.append(np.random.exponential(1/λw[df["CloudType"].values[i]]))
        zm = np.array(zm).reshape(-1,1)
        zw = np.array(zw).reshape(-1,1)

        randm = np.random.rand(len(zm))
        randw = np.random.rand(len(zw))

        bm = np.zeros(len(zm))
        bw = np.zeros(len(zw))
        for i in range(0,len(zm)):
            if randm[i]>1-pm[df["CloudType"][i]]:
                bm[i] = 1
            if randm[i]>1-pw[df["CloudType"][i]]:
                bw[i] = 1

        boolean = df["CloudType"].values<2
        η[boolean] = self.trunc_gauss(0,df.generation[boolean],df.generation[boolean],σ,sum(boolean))

        generated_ts = df.generation.values.reshape(-1,1) + matmul_toeplitz((abs(first_col1), abs(first_row1)), (bm.reshape(-1,1)*zm)) - \
            matmul_toeplitz((first_col2, first_row2), ((bw.reshape(-1,1)*zw))) +η.reshape(-1,1)
        ts["HighRes"] = 0.0
        ts.loc[df.index,"HighRes"] = generated_ts.T[0]
        ts.HighRes[ts.HighRes<0] = 0
        ts.HighRes *= self.capacity/7.5

        return pd.DataFrame(ts["HighRes"].resample(resolution).mean())

            
            
    def trunc_gauss(self,a,b,mu,sigma,N):
        u = np.random.rand(N)
        alpha, beta = (a - mu)/sqrt(sigma), (b - mu)/sqrt(sigma)
        before_inversion = norm.cdf(alpha) + u*(norm.cdf(beta)-norm.cdf(alpha))
        x = norm.ppf(before_inversion)*sqrt(sigma)

        return x
        
         
        
