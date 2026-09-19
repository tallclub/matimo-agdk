// Conventional Commits, same type list as the sibling `matimo` repo.
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore', 'ci', 'revert', 'example'],
    ],
    'subject-case': [0],
  },
};
