# Hierarchical Structure

```
linear_1d

  linear_1d_core    (linear 1d mac operation)
  linear_1d_control (linear 1d controller - generate control signal)
  linear_1d_csr     (linear 1d control register)

  linear_1d_sync_init             ()
  linear_1d_sync_profile_init     ()
  linear_1d_sync_profile_snapshot ()
  linear_1d_sync_bias_go          ()
  linear_1d_sync_input_go         ()
  linear_1d_sync_weight_go        ()
  linear_1d_sync_result_go        ()

  linear_1d_sync_ready
  linear_1d_sync_profile_done
  linear_1d_sync_bias_done
  linear_1d_sync_input_done
  linear_1d_sync_weight_done
  linear_1d_sync_result_done


```

