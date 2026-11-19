# Study design (written before any outcome relationship in the data was examined)

This file fixes the analysis choices before results are seen. Only the structure of the event files (field
names, possession ids, how kick-offs and period ends are coded) was inspected before it was written. The same
StatsBomb files were used earlier in a different project (ball-progression ratings), which looked at action values,
minutes and goal totals; no relationship between the timing of chances, goals and later chances was examined there. Any later
deviation is listed at the bottom with its reason.

## Question
Football talk is full of momentum: a team "on top" keeps creating chances, and a team is "most vulnerable just
after it has scored". Once team strength, the score and the clock are accounted for, is there a short-lived rise
in a team's rate of chances after (a) it creates a chance, (b) it scores, (c) it concedes? How large is it, how
long does it last, and does it improve predictions of matches the model has never seen?

## Data
StatsBomb Open Data, commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`, 2015/16 Premier League, La Liga,
Serie A and Ligue 1: 1,517 matches (the same verified files as the progression-ratings project; every file is
checked against its git blob hash). Bundesliga 2015/16 excluded (only one team's matches are released).

## Units and clock
- A window is one team in one period of one match. Time is the StatsBomb timestamp within the period (it
  restarts at zero each period) and runs to the period's Half End event (stoppage time included).
- History does not carry across half-time for the short windows below.
- Season halves per league, as before: matches before the date of the first round-20 fixture are H1, the rest H2.

## Events
- **Chance (primary event):** a possession in which the team takes at least one shot. All shots by the same team
  in the same possession count as one chance, timed at the first of them. All shot types are included
  (open play, free kicks, corners, penalties).
- **Goal:** a shot with outcome Goal, or an own goal credited to the benefiting team. Teams are always matched by
  team id, never by name.
- **Open-play chance (sensitivity):** a chance whose first shot is of type Open Play and whose possession did not
  start from a corner or free kick (play pattern not "From Corner" / "From Free Kick").
- **Box entry (sensitivity, more events):** a possession in which the team completes a pass or carry ending in the
  penalty area (x >= 102, 18 <= y <= 62 on StatsBomb's 120 x 80 pitch) or shoots from inside it, timed at the
  first such action.
- **Positive control:** individual shots instead of possessions. Rebounds and follow-up shots within the same
  attack must show up as strong excitation within the first two minutes; if the method could not see that, it
  could not see anything.

## Exposure (time at risk)
- From the start of each period to its end, except dead time after goals: from the goal to the following
  kick-off pass in the same period (to the period end if there is none). No team can create a chance during
  that interval; counting it as time at risk would manufacture a fake post-goal dip.
- Injury stoppages and other interruptions are not removed (their duration is not recorded reliably).

## Model: a log-linear self-exciting point process
The intensity of chances for team i at time t is

    log lambda_i(t) = baseline_i(t) + sum over history windows w of gamma_w * count_w(t)

with every covariate constant between change points. The likelihood is then exactly that of a Poisson
regression on time segments with log-duration offsets (piecewise-exponential model), so the fit and the
log-likelihood are exact, with no numerical integration and no simulation.

Baseline terms (all known at time t, nothing from later):
- period x clock bin: [0,15), [15,30), [30,45), [45, end) minutes of each period (8 levels);
- game state from team i's perspective: <= -2, -1, 0, +1, >= +2;
- player advantage: -1, 0, +1 (capped), from the players actually on the pitch (starting XI, substitutions,
  dismissals, as in the progression-ratings project);
- home indicator;
- pre-match strength: log of team i's shrunken chances created per 90 and log of the opponent's shrunken chances
  conceded per 90, each computed from matches played strictly before this match's date (shrinkage: 5 matches'
  worth of the league average of those earlier matches; 0 on a league's first match date).

History terms (counts):
- own chances in the windows (0,2], (2,5], (5,10], (10,20] minutes before t, same period;
- opponent chances in the same windows;
- own and opponent chances earlier in the match (more than 20 minutes before t, or in the first period when t is
  in the second). These absorb persistent match-level intensity learned during the game (a team simply having a
  good day). A momentum effect is a short-window effect larger than this persistent one;
- goals scored by team i in (0,2], (2,5], (5,10], (10,20] minutes before t, same period, and goals conceded in
  the same windows. Because a goal is also a chance (except own goals), the total change after scoring is the
  chance term plus the goal term.

## Estimands fixed in advance (primary set, Holm-corrected together)
1. Self-excitation: gamma(own chance, 0-2 min) minus gamma(own chance, earlier in match).
2. Cross-excitation: gamma(opponent chance, 0-2 min) minus gamma(opponent chance, earlier in match).
3. Scorer after scoring: total log rate ratio for the scoring team in (0,2], (2,5], (5,10]
   (own-chance term plus goal-scored term, each window).
4. Conceder after conceding ("most vulnerable after scoring", seen from the other side): total log rate ratio for
   the conceding team in (0,2], (2,5], (5,10] (opponent-chance term plus goal-conceded term).
Eight tests in total. Reported as rate ratios with 95% intervals.

## Heterogeneity: why three models
Momentum estimates are easily faked by heterogeneity (teams and matches that produce many chances also produce
them close together) and easily hidden by over-correction.
- Main model: pre-match strength + the "earlier in the match" counts (in-game heterogeneity).
- No-heterogeneity model: pre-match strength only, no earlier-in-match counts. Biased towards momentum.
- Fixed-effects model: one intercept per team-match, no pre-match strength or home terms. With about ten
  chances per team-match, fixed effects bias the history coefficients downwards (the dynamic-panel, or Nickell,
  bias). Biased against momentum.
A conclusion is called robust only if it holds in the main and the fixed-effects model.

## Inference
Cluster-robust (sandwich) standard errors with clusters = matches, from the full-season fit. Rate ratios with 95%
intervals. Holm correction over the eight primary tests.

## Out-of-sample validation (temporal)
- Fit on H1 matches only; evaluate on H2 matches the exact point-process log-likelihood of:
  M0 baseline only, M1 + chance history, M2 + goal windows.
- Report the H2 log-likelihood gain per match of M1 over M0 and M2 over M1, with 95% intervals by resampling H2
  matches (whole real matches, with replacement).
- Goodness of fit on H2 by the time-rescaling theorem: transformed inter-event times should be Exp(1); report the
  KS distance and a QQ plot for M0 and M2.
- Observed vs expected chances by minute after a goal on H2, with expectations from the H1-fitted M1 (no goal
  terms): shows directly whether goal windows are needed.

## Sensitivities
Open-play chances; box entries; each league separately (main model); fixed-effects and no-heterogeneity models.

## Deviations from this design
1. **Goodness-of-fit test corrected (an error in this design, found when the first run returned a mean rescaled
   gap of 0.83).** The design used only complete gaps between events. Dropping each window's last, censored gap
   selects short gaps, so even a perfect model would fail the test. Replaced by (a) a Kaplan-Meier estimate of
   the rescaled-gap distribution with the last gap right-censored, compared with Exp(1), and (b) the exact
   conditional test: given n events in a window, Lambda(t_i) / Lambda(T) are iid Uniform(0, 1). Nothing else in
   the analysis changed.

## Corrections made after the first write-up
- Model-free hazard after a chance (exploratory): the first version excluded a chance only when its first shot was
  the goal, so 593 gaps that contained a goal (scored with a later shot of the same attack, or an own goal), and
  therefore a kick-off restart, were kept. All gaps containing a goal are now excluded. The curve moved by at most
  about 3%; no stated number changed.

## Exploratory analyses (decided after seeing the primary results; labelled as such everywhere)
The first run showed that the 0-2 minute windows are dominated by a fall in chances for both teams. Two things
built into the event definition and the game explain this: a chance is a possession, so a follow-up shot in the
same attack can never be a new chance (the shot-level positive control shows the opposite sign, +19%), and after
most chances the ball goes dead (goal kick, corner, keeper). The pre-registered estimand 1 (0-2 min vs earlier)
therefore measures mostly this, not momentum. Added, and reported as exploratory:
- contrasts of each later window (2-5, 5-10, 10-20 min) against the earlier-in-match term, main and
  fixed-effects models, all event types;
- out-of-sample (H1 -> H2) split of the history gain into the 0-2 minute terms and the 2-20 minute terms;
- a model-free view of the seconds after a chance: rate of the next chance (either team) by time since the last
  one, relative to the overall rate;
- standard errors with clusters = teams and two-way (team and opponent), as a robustness check of the
  pre-registered match clustering.
