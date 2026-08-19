## Summary

Describe the problem and the user-visible outcome.

## Verification

- [ ] Production build passes (`npm run build`)
- [ ] Frontend tests pass (`npm test`)
- [ ] Backend tests pass (`python -m unittest discover -s backend/tests -v`) or are not affected
- [ ] UI changes include non-sensitive screenshots or a short recording

## Model, data, and license impact

- [ ] No model, dataset, preprocessing, calibration, or license contract changes
- [ ] Any such changes are documented with immutable revisions and hashes

## Safety and privacy

- [ ] No credentials, `.env.local` files, private media, model weights, runtime outputs, or personal filesystem paths are included
- [ ] TRIBE output remains separate from behavioral or clinical claims
- [ ] New behavioral metrics fail closed unless their validation contract is satisfied
