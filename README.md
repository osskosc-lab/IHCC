# IHCC v1.1 Phase 1

介入的履歴因果円錐（Interventional History Causal Cone; IHCC）の、合成系に限定した境界回復・誤指定・識別限界監査です。

主命題は一つです。

> 指定した有限介入集合、状態クラス、TV閾値のもとで、推定器が埋め込み済みの包含最小履歴因果基底を回復できるか。

主要採用条件は `F1_basis >= 0.80`、非刈込み族に対する `delta F1 >= 0.10`、G1--G7とM1--M4の規定判定です。Phase 1が判定するのは合成系内の推定仕様だけであり、表現不変の非Markov実体、量子機構、Orch OR、意識を支持しません。

## 実行

```bash
python -m pip install -e .
python run_experiment.py --profile smoke --output-dir results/smoke
python run_experiment.py --profile pilot --output-dir results/pilot
python run_experiment.py --profile confirmatory --output-dir results/confirmatory
```

`smoke`は配線確認、`pilot`は30シードの暫定監査、`confirmatory`だけが事前登録済み100シード判定です。科学ゲートの不通過はプログラム異常とは限らないため、実験コマンドは結果を保存して正常終了します。実装例外・非有限値・スキーマ破損は終了コード2になります。

GitHub Actionsの確証実験は100シードを10 shardへ分割します。各shardは`--seed-offset`と`--seed-count`で部分実行され、単独では確証判定を出しません。全shardのシード集合が事前登録済み100シードと完全一致した場合だけ、`aggregate_results.py`がG1--G7/M1--M4を再計算します。

## 実装したシナリオ

| ID | 正解として固定した判定 |
|---:|---|
| 0 | 完全ヌル、空基底 |
| 1 | 疎な単独基底 |
| 2 | 積パリティによる純粋二変数相乗基底 |
| 3 | 単独基底と冗長上位集合の分離 |
| 4 | 三成分有限十分状態、完全リセットで残差消失 |
| 5 | 状態欠落・不完全リセット由来の偽残差 |
| 6 | 有限指数状態クラス外の冪記憶残差 |
| 7 | 代数的非可換性と操作的順序効果の正負対照 |
| 8 | collider条件付けと因果リセットの乖離 |
| 9 | 同一有限応答表を持つ量子様媒介／文脈依存古典ミミックの識別不能 |

## 重要な統計契約

- 最大介入対の選択foldと評価foldを分離します。
- 主要TV推定器はcalibration付きgradient boostingです。これはモデル族相対の下限として扱います。
- 各外側学習foldの内部を75%学習／25%Platt校正へ固定分割し、評価foldは介入対選択にも校正にも使用しません。
- 136候補に対する同時下限は、1000 bootstrapでは尾部分位点を解像できないため、片側Clopper--Pearson区間をBonferroni配分して構成します。
- bootstrapは候補別95%区間、permutationは監査p値に使い、主選択の同時下限とは混同しません。
- `{-1,+1}`上のXORは積パリティ `U1 * U2` として実装します。
- 最大次数2では相乗ペアに冗長変数を加えた3変数集合を探索できないため、Scenario 3は単独効果の冗長上位集合を正対照にします。

詳細は [事前登録仕様](docs/preregistration.md) と [統計契約](docs/statistical_contract.md) を参照してください。

## 出力

- `seed_level_icq.csv`
- `estimated_minimal_bases.json`
- `basis_recovery_summary.csv`
- `point_cone_summary.csv`
- `reset_residual_summary.csv`
- `tv_estimator_audit.csv`
- `noncommutativity_audit.csv`
- `conditional_reset_audit.csv`
- `mimic_equivalence.csv`
- `falsification_gates.csv`
- `preregistered_results.md`
- `ihcc_v11_report.pdf`
