import datajoint as dj
import spyglass.spikesorting.v1 as sgs
import random
import multiprocessing as mp

def worker(table_name, chunk_keys, populate_kwargs = None, curation_kwargs = None):
    '''
    table_name: spyglass table name as sgs.{table_name}
    
    '''

    dj.conn(reset=True)
    
    if table_name == sgs.CurationV1:
        
        for key in chunk_keys:
            try:
                (table_name).insert_curation(key, **curation_kwargs)
            except Exception as e:
                print("Failed:", key, e)
    
    
    
    else:
        for key in chunk_keys:     
            try:   
                (table_name).populate(key, **populate_kwargs)
            except Exception as e:
                print("Failed:", key, e)



def launch_workers(table_name, keys, n_workers=6, curation_kwargs = None):
    try:
        dj.conn().close()
    except Exception:
        pass
    populate_kwargs = dict(
        use_transaction=True,
        reserve_jobs=False, #TODO; should be True to avoid duplicate jobs, but prevents jobs lost through stalled workers from completing
        suppress_errors=False,
        order="random",
        display_progress=True,
    )
    ctx = mp.get_context("spawn")  
    
    seed = 42
    random.Random(seed).shuffle(keys)
    
    num_chunks = max(1, min(n_workers, len(keys)))
    chunks = [keys[i::num_chunks] for i in range(num_chunks)]
    
    if table_name == sgs.CurationV1:
        procs = [ctx.Process(target=worker, args=(table_name, chunk, curation_kwargs)) for chunk in chunks if chunk]
    else:
        procs = [ctx.Process(target=worker, args=(table_name, chunk, populate_kwargs)) for chunk in chunks if chunk]
    for p in procs: p.start()
    return procs

















