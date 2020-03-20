import geoip2.database
reader = geoip2.database.Reader('geolite2/GeoLite2-City_20200107/GeoLite2-City.mmdb')


from ipwhois import IPWhois
from pprint import pprint

import sys
name=sys.argv[1]
response = reader.city(name)
obj = IPWhois(name)

results = obj.lookup_whois()
ISP=results['asn_description'].split('-')[0]


from math import sin, cos, sqrt, atan2, radians
lat1=radians(13.002789)
lon1=radians(77.596464)
from math import sin, cos, sqrt, atan2
lat2=radians(response.location.latitude)
lon2=radians(response.location.longitude)
R = 6373.0

dlon = lon2 - lon1
dlat = lat2 - lat1
a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
c = 2 * atan2(sqrt(a), sqrt(1-a))
distance = R * c

print("ISP:",ISP)
print("Distance from Deeproot Office:",distance)
print("City:",response.city.name)
print("Country:",response.country.name)
print("Sub-Division:",response.subdivisions.most_specific.name)
print("Postal Code:",response.postal.code)
print("Latitude:",response.location.latitude)
print("Longitude:",response.location.longitude)






