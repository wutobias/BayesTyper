
def retrieve_failed_workers(worker_id_list, verbose=False):

    from ray.util import state

    failed_keywords = [
        "fail",
        "failed",
        "error",
        "nil",
        "pending_args_avail",
        "pending_node_assignment",
    ]

    succes_keywords = [
        "finish",
        "running",
    ]

    failed_worker_id_list = list()
    for worker_id in worker_id_list:
        failed = False
        try:
            s = state.get_task(
                worker_id.task_id().hex()
                )
        except:
            s = None
        if s == None:
            failed = True
        elif isinstance(s, list):
            for _s in s:
                if failed:
                    break
                try:
                    _state =_s.state.lower().rstrip().lstrip()
                except:
                    failed = True
                    break
                for k in failed_keywords:
                    if k in _state:
                        failed = True
                        if verbose:
                            print(
                                f"Failed due to {_state}"
                                )
                        break

                if verbose and not failed:
                    _found = False
                    for k in succes_keywords:
                        if k in _state:
                            _found = True
                            break
                    if not _found:
                        print(
                            "WHAT IS THIS??????????",
                            _state,
                        )
        else:
            try:
                _state = s.state.lower().rstrip().lstrip()
                for k in failed_keywords:
                    if k in _state:
                        failed = True
                        if verbose:
                            print(
                                f"Failed due to {_state}"
                                )
                        break
            except:
                failed = True

            if verbose and not failed:
                _found = False
                for k in succes_keywords:
                    if k in _state:
                        if verbose:
                            print(
                                f"Failed due to {_state}"
                                )
                        _found = True
                        break
                if not _found:
                    print(
                        "WHAT IS THIS??????????",
                        _state,
                    )

        if failed:
            failed_worker_id_list.append(
                worker_id
                )

    return failed_worker_id_list