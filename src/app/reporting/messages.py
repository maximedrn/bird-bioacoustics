"""Presentation text for plots, notebook observations and coverage."""

from typing import Final


class PlotLabel:
    """Centralize graph labels independently of scientific measurements."""

    RECORDINGS_IN_THE_CATALOGUE_SNAPSHOT: Final[str] = (
        "Recordings in the catalogue snapshot"
    )
    ANALYSIS_OF_CACHED_RECORDINGS: Final[str] = "Analysis of cached recordings"
    CATALOGUED_RECORDINGS: Final[str] = "Catalogued recordings"
    TARGET_SPECIES_CONFIDENCE: Final[str] = "Target-species confidence"
    NOISE_TARGET_CONFIDENCE: Final[str] = "Noise: target confidence"
    TARGET_DETECTED_IN_EXPORTED_PREDICTIONS: Final[str] = (
        "Target detected in exported predictions"
    )
    NOISE_TARGET_DETECTION_RATE: Final[str] = "Noise: target detection rate"
    A_B_MIX_AMPLITUDE_COEFFICIENTS: Final[str] = (
        "A/B mix (amplitude coefficients)"
    )
    BIRDNET_CONFIDENCE: Final[str] = "BirdNET confidence"
    EFFECT_OF_OVERLAPPING_BIRD_VOCALIZATIONS: Final[str] = (
        "Effect of overlapping bird vocalizations"
    )
    COUNT_ACROSS_RECORDINGS: Final[str] = "Count across recordings"
    RETAINED_PREDICTIONS: Final[str] = "Retained predictions"
    RECORDING_LEVEL_RATE: Final[str] = "Recording-level rate"
    TARGET_RETENTION_OPTIONAL_REFERENCE_METRICS: Final[str] = (
        "Target retention / optional reference metrics"
    )
    TIME_S: Final[str] = "Time (s)"
    FREQUENCY_HZ: Final[str] = "Frequency (Hz)"
    TARGET_SEGMENT_SPECTROGRAM: Final[str] = "Target-segment spectrogram"
    CACHED: Final[str] = "Cached"
    STILL_TO_COLLECT: Final[str] = "Still to collect"
    CATALOGUE_COVERAGE_TOTAL_UNKNOWN: Final[str] = (
        "Catalogue coverage (total unknown)"
    )
    ANALYSED: Final[str] = "Analysed"
    FAILED: Final[str] = "Failed"
    PENDING: Final[str] = "Pending"
    RECORDING_MEANS_1_SD: Final[str] = "Recording means ± 1 SD"
    ORIGINAL_SEGMENTS_MEAN: Final[str] = "Original segments (mean)"
    MEAN_DETECTION_RATE: Final[str] = "Mean detection rate"
    ORIGINAL_SEGMENTS: Final[str] = "Original segments"
    SNR_DB_LOWER_VALUES_MEAN_STRONGER_NOISE: Final[str] = (
        "SNR (dB) — lower values mean stronger noise"
    )
    TARGET_A_MEAN_1_SD: Final[str] = "Target A (mean ± 1 SD)"
    TARGET_B_MEAN_1_SD: Final[str] = "Target B (mean ± 1 SD)"
    DETECTIONS: Final[str] = "Detections"
    DISTINCT_SPECIES: Final[str] = "Distinct species"
    METADATA_TARGET_RECOVERED: Final[str] = "Metadata target recovered"
    CONFIDENCE_THRESHOLD: Final[str] = "Confidence threshold"
    POWER_DB: Final[str] = "Power (dB)"
    CATALOGUE_COVERAGE: Final[str] = "Catalogue coverage ({ratio:.1%})"


class ObservationMessage:
    """Describe cumulative results using explicit template fields."""

    HEADING: Final[str] = "### Observation"
    NOISE_BASELINE: Final[str] = (
        "Les segments de **{count} enregistrements** ont été testés. La "
        "confiance moyenne sans bruit ajouté est "
        "**{baseline_confidence:.3f}**."
    )
    NOISE_LEVEL: Final[str] = (
        "- **{snr_db:g} dB** : confiance moyenne "
        "**{confidence_mean:.3f}**, écart-type entre enregistrements "
        "**{confidence_std:.3f}**, cible retrouvée dans "
        "**{detection_rate:.1%}** des essais."
    )
    NOISE_EXPLANATION: Final[str] = (
        "Les barres montrent un écart-type des moyennes par "
        "enregistrement. La variabilité entre réalisations de bruit est "
        "exportée séparément dans la table `noise_summaries` du point de "
        "reprise SQLite. Une valeur 0 indique une cible absente des "
        "prédictions exportées au seuil minimal de 0,10."
    )
    NOISE_SKIPPED: Final[str] = (
        "**{skipped} enregistrements** sans cible détectée au départ ont "
        "été exclus de ce test et sont listés dans la colonne "
        "`noise_skip_reason` du catalogue SQLite."
    )
    PARAGRAPH_BREAK: Final[str] = "\n\n"
    LINE_BREAK: Final[str] = "\n"
    THRESHOLD_COUNTS: Final[str] = (
        "### Observation\n\nEntre les seuils **{first_threshold:.2f}** et "
        "**{last_threshold:.2f}**, les détections conservées passent de "
        "**{first_detections}** à **{last_detections}**, et les espèces "
        "prédites de **{first_species}** à **{last_species}**.\n\n"
    )
    TARGET_RECOVERY: Final[str] = (
        "L'espèce principale indiquée sur Xeno-canto est retrouvée dans "
        "**{first_rate:.1%}** puis **{last_rate:.1%}** des enregistrements "
        "dont la cible est identifiée et couverte par BirdNET. Ce taux "
        "décrit la récupération de la cible connue, pas le rappel de "
        "toutes les espèces."
    )
    NO_TARGET: Final[str] = (
        "Aucune cible identifiée couverte par BirdNET n'est disponible "
        "dans les enregistrements traités."
    )
    COMPLETE_ANNOTATIONS: Final[str] = (
        "\n\nLa précision, le rappel et le F1 micro portent uniquement sur "
        "les **{recordings} enregistrements complètement annotés**, en "
        "comptant chaque couple enregistrement/espèce une seule fois."
    )
    NO_ANNOTATIONS: Final[str] = (
        "\n\nSans annotations complètes, précision, rappel et F1 restent non "
        "calculés."
    )
    OVERLAP: Final[str] = (
        "### Observation\n\nLes moyennes portent sur **{pair_count} paires** "
        "d'enregistrements distincts. Entre les mélanges {first_mix} et "
        "{last_mix}, la confiance moyenne de la cible A passe de "
        "**{first_a:.3f}** à **{last_a:.3f}**, et celle de la cible B de "
        "**{first_b:.3f}** à **{last_b:.3f}**.\n\nA et B désignent les rôles "
        "dans chaque paire ; les espèces exactes figurent dans la table "
        "`overlap` du point de reprise SQLite. Les barres montrent un "
        "écart-type entre paires. Les coefficients portent sur "
        "l'amplitude, sans égalisation préalable des niveaux."
    )


class ReportMessage:
    """Present saved corpus coverage and unavailable figures."""

    COVERAGE: Final[str] = (
        "**{processed:,} audios analysés** · **{failed:,} échecs** · "
        "**{batches} lots terminés** · **curseur {cursor:,}**"
    )
    MISSING_MEASUREMENTS: Final[str] = (
        "Les mesures disponibles ne permettent pas encore de tracer ce "
        "graphique."
    )
