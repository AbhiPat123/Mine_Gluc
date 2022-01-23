# run DBSCAN for different epsilon and min_samples
gt_point = np.array(np.sort(np.unique(mdGT, return_counts=True)[1]))
min_dist_params = []
min_dist_GT = np.inf
eps_range = range(5,eps_max)
ms_range = range(5,mpts_max)
if False:
    for eps_v in eps_range:
        for ms_v in ms_range:
            mod_DB = DBSCAN(eps=eps_v, min_samples=ms_v)
            fDB= mod_DB.fit( mdFT )
            lDB = fDB.labels_
            if np.max(lDB) == -1:
                continue
            cnl = np.where(lDB!=-1)
            cl_DBKM = bis_KM_till_k(mdFT_GT[cnl], lDB[cnl], req_num_clusters)
            if cl_DBKM is None:
                continue
            dbkm_point = np.array(np.sort([ len(clst) for clst in cl_DBKM ]))

            dist = np.linalg.norm(gt_point-dbkm_point)

            if min_dist_GT > dist:
                min_dist_GT = dist
                min_dist_params = []

            if min_dist_GT == dist:
                min_dist_params.append([eps_v, ms_v])
    print("GOOD VALUES:")
    print(min_dist_params)

    with open('params.pkl', 'wb') as p_file:
        pickle.dump(min_dist_params, p_file)


#load params
#with open('params.pkl', 'rb') as params_file:
    #params = pickle.load(params_file)
#params