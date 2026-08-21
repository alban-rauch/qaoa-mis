# Conditions studied

## 🔧 Usage

```
# For condX:
path = COND_DIR / "condX.json"
problem_config, strategy_config, apparatus_config = source.utils.cond_gen.load_condition(path)
```

## 📄 Conditions list

<style>

  .soft-table {
    border-collapse: collapse;
    width: 100%;
  }

  .soft-table th, 
  .soft-table td {
    border-right: 0.5px solid #fcfcfca4; 
    padding: 2px 2px;
    text-align: center;
    vertical-align: middle;
  }

  .soft-table th:last-child, 
  .soft-table td:last-child {
    border-right: none;
  }

  .soft-table thead tr {
    border-bottom: 1.5px solid #fcfcfca4;
  }
  
  .soft-table tbody tr {
    border-bottom: 1px solid #fcfcfca4;
  }

  .soft-table th:first-child, 
  .soft-table td:first-child:not([colspan]) { color: #0088d1;
  font-weight: bold }
  .cond1 { color: #00f6ee; }
  .cond2 { color: #00f6b9; }
  .cond3 { color: #00f610; }
  .cond4 { color: #98f600; }

  .config-row td {
    border-top: 1.5px solid #fcfcfca4;
    border-bottom: 1px solid #fcfcfca4;
    border-right: none;
    font-weight: bold;
  }
  
  .change {
    color: #c25ed8; 
    font-weight: bold;
  }

</style>

<table class="soft-table">
  <thead>
    <tr>
      <th class="var">Variable</th>
      <th class="cond1">cond1</th>
      <th class="cond2">cond2</th>
      <th class="cond3">cond3</th>
      <th class="cond4">cond4</th>
    </tr>
  </thread>
  <tbody>
    <tr class="config-row">
      <td colspan="5"><b>Problem</b></td>
    </tr>
    <tr>
      <td>N</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td>Graph</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr class="config-row">
      <td colspan="5"><b>Strategy</b></td>
    </tr>
    <tr>
      <td>constrained</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
      <td>❌</td>
    </tr>
    <tr class="change">
      <td>Relaxation type</td>
      <td>❌</td>
      <td>continuous</td>
      <td>continuous</td>
      <td>continuous</td>
    </tr>
    <tr class="change">
      <td>Parameter transfer</td>
      <td>INTERP</td>
      <td>INTERP</td>
      <td>FOURIER [5, 10]</td>
      <td>INTERP</td>
    </tr>
    <tr class="change">
      <td>Mixers<br>(init params)</td>
      <td>X<br>(0.67, 0.33)</td>
      <td>X<br>(0.67, 0.33)</td>
      <td>X<br>(0.67, 0.33)</td>
      <td>X & Y<br>(0.67, 0.33, 0.33)</td>
    </tr>
    <tr class="config-row">
      <td colspan="5"><b>Apparatus</b></td>
    </tr>
    <tr>
      <td>p</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
      <td>—</td>
    </tr>
    <tr>
      <td>Estimator diff method</td>
      <td>Adjoint</td>
      <td>Adjoint</td>
      <td>Adjoint</td>
      <td>Adjoint</td>
    </tr>
    <tr>
      <td>Sampler shots</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000</td>
    </tr>
    <tr>
      <td>Optimizer<br>(max steps)</td>
      <td>L-BFGS-B<br>(1000)</td>
      <td>L-BFGS-B<br>(1000)</td>
      <td>L-BFGS-B<br>(1000)</td>
      <td>L-BFGS-B<br>(1000)</td>
    </tr>
  </tbody>
</table>


## ⏳ Old case

```
problem_config = {
    "N": None,
    "graph": None, # gph.randomDRegular(N, 3) | gph.randomGilbert(N, 0.25)
}

strategy_config = {
    "constrained": False,
    "relaxation_type": 'continuous',    #  None | 'continuous'
    "param_transfer_type": 'interp',    # 'given' | 'random' | 'interp' | 'fourier'
    "fourier_qR": [None, 5],
    "init_param": [0.67, 0.33],
    "mixers": ["x"],
}

apparatus_config = {
    "p": None,
    "device": "lightning.amdgpu",        # "lightning.qubit" | "lightning.amdgpu" 
    "estimator_shots": 10000,
    "sampler_shots": 10000,
    "optimizer": "L-BFGS-B",            # "L-BFGS-B" | "Adam"
    "opt_steps": 1000,
}
```