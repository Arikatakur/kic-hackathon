add animations to the html, I dont want alot of empty spaces, add example images, for example one for the "entomological LiDAR", also add sources for my thesis,

Best research to include
Part of your idea	Research that supports it	How to use it
Laser/optical sensing can detect flying insects	Rydhmer et al., Automating insect monitoring using unsupervised near-infrared sensors	Shows that compact near-infrared sensors can record insect flight signals at kHz speed and detect wingbeat harmonics, melanisation, and flight direction. Their sensors recorded about 19× more insect observations than yellow water traps.
Wingbeat + optical fingerprint is useful for classification	Chen et al., Flying Insect Classification with Inexpensive Sensors	Supports using optical sensors instead of only acoustic sensors. Important warning: do not rely only on wingbeat frequency; use multiple features.
Entomological LiDAR can monitor insects in flight	Bernenko et al., Insect Diversity Estimation in Polarimetric Lidar	Directly supports the phrase entomological LiDAR. They used LiDAR to observe wild flying insects and extract wingbeat frequency, daily activity, and spatial distribution. It also says LiDAR can differentiate many “signal types,” but exact species ID is still limited.
Multisensor fusion is better than one sensor	Tschaikner et al., Multisensor Data Fusion for Automatized Insect Monitoring (KInsecta)	Very strong source for your architecture: camera + optical wingbeat sensor + environmental sensors + AI fusion. It describes a low-cost system tested in lab and field.
Camera confirmation is valid	Brandt et al., Low Cost Machine Vision for Insect Classification	Supports your macro-camera part. They built a low-cost imaging system with controlled lighting and reduced motion blur, reaching >96% classification accuracy on 16 insect species.
AI pest recognition is already a real research direction	Ung et al., An Efficient Insect Pest Classification Using Multiple CNN-Based Models	Supports the AI image-classification part. It explains why manual pest recognition is slow and expensive, and why CNNs are used for pest classification.
Asian citrus psyllid is a strong target pest	UC Riverside Center for Invasive Species Research	Supports choosing Asian citrus psyllid: it is an efficient vector of HLB/citrus greening, one of the most destructive citrus diseases, and there is no cure.
Laser + camera + wingbeat has real precedent	Photonic Fence / Wired report	Not academic research, but useful as industry precedent. It describes a system using cameras/optics to assess insect form, velocity, acceleration, and wingbeat frequency, with Asian citrus psyllid as an agricultural target.
Best sources to cite in your presentation

Use these 5 as your main evidence:

Rydhmer et al. 2021 — near-infrared insect monitoring.
Use for: “optical sensors can monitor flying insects automatically.”
Bernenko et al. 2024 — entomological/polarimetric LiDAR.
Use for: “LiDAR-like photonic sensing can detect insect flight signatures.”
Tschaikner et al. 2024, KInsecta — multisensor fusion.
Use for: “camera + wingbeat + environmental data is a valid architecture.”
Brandt et al. 2024 — low-cost machine vision.
Use for: “controlled macro imaging can classify insects accurately.”
UC Riverside ACP page — biological problem.
Use for: “Asian citrus psyllid is a dangerous citrus disease vector.”
Thesis / dissertation

I found one relevant historical thesis, but it is more of a biological foundation than a modern AI/laser system:

Carroll M. Williams, 1941, Harvard PhD thesis
Title: A morphological and physiological analysis of the flight of Drosophila, with special reference to the factors controlling the frequency of wingbeat.
Use it only if you want to show that wingbeat frequency has been studied scientifically for a long time. Do not use it as proof of modern LiDAR detection.

For the actual hackathon, the modern papers above are stronger than a thesis.

Sentence you can put in the presentation

Previous studies show that flying insects can be monitored using near-infrared optical sensing, wingbeat harmonics, and entomological LiDAR. Recent multisensor systems combine optical wingbeat sensors, controlled camera imaging, environmental sensors, and AI-based data fusion for insect classification. Based on this research, our solution uses laser-based insect sensing to trigger close-range imaging and classify events as target pest, non-target insect, or unknown.

Important claim to avoid

Do not say:

LiDAR gives high-resolution images of insects.

Say:

Laser-based insect sensing extracts an optical flight fingerprint, while the camera provides visual confirmation.

and here's the links for each: Here are the links to add:

#	Source	Link
1	Automating insect monitoring using unsupervised near-infrared sensors — supports optical insect detection, wingbeat harmonics, NIR sensing	https://arxiv.org/abs/2108.05435
2	Flying Insect Classification with Inexpensive Sensors — supports optical sensors + wingbeat features for insect classification	https://arxiv.org/abs/1403.2654
3	Insect Diversity Estimation in Polarimetric Lidar — supports entomological LiDAR / photonic insect sensing	https://arxiv.org/abs/2406.01143
4	Multisensor Data Fusion for Automatized Insect Monitoring (KInsecta) — supports combining camera + optical wingbeat sensor + environmental sensors	https://arxiv.org/abs/2404.18504
5	Low Cost Machine Vision for Insect Classification — supports controlled imaging / macro camera classification	https://arxiv.org/abs/2404.17488
6	An Efficient Insect Pest Classification Using Multiple CNN-Based Models — supports AI/CNN pest classification	https://arxiv.org/abs/2107.12189
7	UC Riverside — Asian Citrus Psyllid — supports why Asian citrus psyllid is a dangerous disease vector for citrus	https://cisr.ucr.edu/invasive-species/asian-citrus-psyllid
8	Photonic Fence / Wired — industry precedent for camera + optics + wingbeat-based insect targeting, including Asian citrus psyllid	https://www.wired.com/story/intellectual-ventures-bug-zapper
9	Carroll M. Williams PhD thesis, Harvard, 1941 — historical thesis on wingbeat frequency in Drosophila	http://id.lib.harvard.edu/aleph/004011783/catalog

Use 1–6 as the main research support.
Use 7 for the biological/agricultural importance.
Use 8 only as a real-world example, not as academic proof.
Use 9 only if you specifically want to mention that wingbeat-frequency research has existed for decades.

you can design it as you see fit