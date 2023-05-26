from det3d.datasets.nuscenes import nusc_common as nu_ds


nu_ds.create_nuscenes_infos("../data/nuScenes", version="v1.0-test", nsweeps=10)
