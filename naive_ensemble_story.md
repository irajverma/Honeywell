# Presentation Script: "Why Not Just Combine All Three Models?"

**Target Length:** ~90 seconds read aloud  
**Tone:** Confident, technical, engineering-driven, conversational  

---

## Spoken Script

> *"When people first look at our architecture, the obvious question is: **'Why didn't you just take a simple weighted average of the baseline rules, Isolation Forest, and the LSTM sequence model?'**"  
>  
> *"That was actually our first attempt! On paper, combining all three model scores linearly seemed like the textbook approach. But when we ran our first evaluation at a realistic SOC budget—the Top 1% alert queue—we hit a major roadblock."*  
>  
> *"The naive linear blend actually **made our security worse**. Core anomaly recall collapsed from **72.4% down to 38.6%**, missing over 220 real attacks that the static baseline alone had caught cleanly. Even worse, lateral movement detection dropped 3-fold!"*  
>  
> *"When we audited the failure, we uncovered a classic machine learning trap: **score crowding and scale mismatch**. Unsupervised sequence models output continuous reconstruction loss. Even slight background fluctuations added small noise scores to thousands of normal events. Those boosted normal events leapfrogged high-confidence baseline policy rules, filling up the top alert queue with noise."*  
>  
> *"That discovery forced us to redesign our ensemble. We built a **Tier-Gated Monotonic Blend**. We established a strict rule floor: any high-confidence baseline rule violation (score \(\ge 3.0\)) is preserved at top priority and never diluted. Then, for remaining events, we apply a sequence boost only when an entity exhibits multi-step resource traversal (\(\text{resources} \ge 5\))."*  
>  
> *"The result? Zero performance regression on static rules, **+6 additional lateral movement catches lifting recall from 44% to 52%**, and an aggregate catch rate of **81.3%** within a tight 2.5% budget. That's why a naive average fails—and why monotonic tiering is essential for real-world SOC operations."*
