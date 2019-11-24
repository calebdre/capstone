import pandas as pd
import numpy as np
import random
import os

def get_ideologies(ideologies_dir):
    ideals1 = pd.read_csv(ideologies_dir + "user-ideal-points-201807-000000000000.csv", usecols=["id_str", "theta"])
    ideals2 = pd.read_csv(ideologies_dir + "user-ideal-points-201807-000000000001.csv",  usecols=["id_str", "theta"])
    ideals3 =  pd.read_csv(ideologies_dir + "user-ideal-points-201807-000000000002.csv",  usecols=["id_str", "theta"])
    
    return ideals1.append(ideals2).append(ideals3)

def get_geolocations(geolocations_dir):
    geos1 = pd.read_csv(geolocations_dir + "already_located_random_users_machine00.csv", usecols=["user_id", "longitude", "latitude", "raw_location", "country"])
    geos2 = pd.read_csv(geolocations_dir + "already_located_random_users_machine01.csv", usecols=["user_id", "longitude", "latitude", "raw_location", "country"])
    geos3 = pd.read_csv(geolocations_dir + "already_located_random_users_machine02.csv", usecols=["user_id", "longitude", "latitude", "raw_location", "country"])

    return geos1.append(geos2).append(geos3)

def get_users(random_users, followers_dir, sample_size):
    result = []
    failures = 0
    for i, user_id in enumerate(random_users):
        try:
            followers = pd.read_csv(followers_dir + str(random_users[i]) + "/2019__10__" + str(random_users[i]) + ".csv")["user_id_followers"].values

            for follower in followers:
                result.append({
                    "node_1": user_id,
                    "node_2": follower
                })
        except Exception as e:
            failures += 1
    
    print(str(failures) + " failures (total: " + str(len(random_users)) + ")")
    return pd.DataFrame(result)
    
def gather_data():
    ideologies_dir = "./"
    geolocations_dir = "./"
    followers_dir = "../followers/"
    output_filename = "followerified.csv"
    sample_size = 150
    
    print("getting ideologies...")
    ideals = get_ideologies(ideologies_dir)
    print("getting geographies...")
    geos = get_geolocations(geolocations_dir)
    
    ideal_ids = ideals["id_str"]
    geo_ids = geos["user_id"]
    user_ids = os.listdir(followers_dir)
    print("finding common users....")
    combined_ids = np.intersect1d(np.intersect1d(ideal_ids.values, geo_ids.values), user_ids)
    
    random_users = random.sample(combined_ids.tolist(), sample_size)
    
    print("assembling graph...")
    graph = get_users(random_users, followers_dir, sample_size)
    
    print("reindexing the graph...")
    r = np.concatenate((graph["node_1"].unique(),graph["node_2"].unique()), axis=0)
    rl = r.tolist()
    graph["node_1"] = graph["node_1"].apply(lambda x: rl.index(x))
    graph["node_2"] = graph["node_2"].apply(lambda x: rl.index(x))
    
    print("creating csv file")
    graph.to_csv(output_filename, index=False)
    
    print("Done!")

if __name__ == "__main__":
    gather_data()