def retrieve_failed_workers(worker_id_list, verbose=False):

    from ray.util import state

    failed_worker_id_list = list()
    for worker_id in worker_id_list:
        failed = False
        s = state.get_task(
            worker_id.task_id().hex()
            )
        if s == None:
            failed = True
        elif isinstance(s, list):
            for _s in s:
                if "fail" in _s.state.lower():
                    failed = True
                elif "error" in _s.state.lower():
                    failed = True
                elif "nil" in _s.state.lower():
                    failed = True
                elif "pending_args_avail" in _s.state.lower():
                    failed = True
                elif "pending_node_assignment" in _s.state.lower():
                    failed = True

                if verbose:
                    if "finish" not in _s.state.lower() and "running" not in _s.state.lower():
                        print(
                            "WHAT IS THIS??????????",
                            _s
                        )
        else:
            if "fail" in s.state.lower():
                failed = True
            elif "error" in s.state.lower():
                failed = True
            elif "nil" in s.state.lower():
                failed = True
            elif "pending_args_avail" in s.state.lower():
                failed = True
            elif "pending_node_assignment" in s.state.lower():
                failed = True

            if verbose:
                if "finish" not in s.state.lower() and "running" not in s.state.lower():
                    print(
                        "WHAT IS THIS??????????",
                        s
                    )
        if failed:
            failed_worker_id_list.append(
                worker_id
                )

    return failed_worker_id_list