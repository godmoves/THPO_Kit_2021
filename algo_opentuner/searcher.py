# coding=utf-8

"""
In opentuner, many search techniques are already available. All the names of
the techniques can be found as follows:
```
>>> import opentuner
>>> techniques, generators = opentuner.search.technique.all_techniques()
>>> for t in techniques:
...     print t.name
```
A user can also create new search techniques
(http://opentuner.org/tutorial/techniques/).

Opentuner will create a multi-arm bandit of multiple techniques if more than
one technique is specified in `args.technique`.

Some bandits with pre-defined techniques are already registered in:
`opentuner.search.bandittechniques`

By default, we use a pre-defined bandit called `'AUCBanditMetaTechniqueA'` of 4
techniques:
```
register(AUCBanditMetaTechnique([
        differentialevolution.DifferentialEvolutionAlt(),
        evolutionarytechniques.UniformGreedyMutation(),
        evolutionarytechniques.NormalGreedyMutation(mutation_rate=0.3),
        simplextechniques.RandomNelderMead()],
        name='AUCBanditMetaTechniqueA'))
```
The other two bandits used in our experiments are: PSO_GA_DE and PSO_GA_Bandit.
Specifying a list of multiple techniques will use a multi-arm bandit over them.
"""
import warnings
from argparse import Namespace

from opentuner.api import TuningRunManager
from opentuner.resultsdb.models import DesiredResult, Result
from opentuner.measurement.interface import DefaultMeasurementInterface as DMI
from opentuner.search.manipulator import ConfigurationManipulator, FloatParameter

# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher


class Searcher(AbstractSearcher):
    searcher_name = "OpentunerSearcher"

    def __init__(self, parameters_config, n_iter, n_suggestion):
        """ Init searcher

        Args:
            parameters_config: parameters configuration, consistent with the definition of parameters_config of EvaluateFunction. dict type:
                    dict key: parameters name, string type
                    dict value: parameters configuration, dict type:
                        "parameter_name": parameter name
                        "parameter_type": parameter type, 1 for double type, and only double type is valid
                        "double_max_value": max value of this parameter
                        "double_min_value": min value of this parameter
                        "double_step": step size
                        "coords": list type, all valid values of this parameter.
                            If the parameter value is not in coords,
                            the closest valid value will be used by the judge program.

                    parameter configuration example, eg:
                    {
                        "p1": {
                            "parameter_name": "p1",
                            "parameter_type": 1
                            "double_max_value": 2.5,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0, 2.5]
                        },
                        "p2": {
                            "parameter_name": "p2",
                            "parameter_type": 1,
                            "double_max_value": 2.0,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0]
                        }
                    }
                    In this example, "2.5" is the upper bound of parameter "p1", and it's also a valid value.

        n_iteration: number of iterations
        n_suggestion: number of suggestions to return
        """
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)

        # Opentuner requires DesiredResult to reference suggestion when making
        # its observation. x_to_dr maps the dict suggestion to DesiredResult.
        self.x_to_dr = {}
        # Keep last suggested x and repeat it whenever opentuner gives up.
        self.dummy_suggest = None

        """Setting up the arguments for opentuner. You can see all possible
        arguments using:
        ```
        >>> import opentuner
        >>> opentuner.default_argparser().parse_args(['-h'])
        ```
        We only change a few arguments (other arguments are set to defaults):
        * database = MEMORY_ONLY_DB: to use an in-memory sqlite database
        * parallelism = n_suggestions: num of suggestions to give in parallel
        * technique = techniques: a list of techniques to be used by opentuner
        * print_params = False: to avoid opentuner from exiting after printing
            param spaces
        """
        DEFAULT_TECHNIQUES = ("AUCBanditMetaTechniqueA",)
        MEMORY_ONLY_DB = "sqlite://"

        args = Namespace(
            bail_threshold=500,
            database=MEMORY_ONLY_DB,
            display_frequency=10,
            generate_bandit_technique=False,
            label=None,
            list_techniques=False,
            machine_class=None,
            no_dups=False,
            parallel_compile=False,
            parallelism=n_suggestion,
            pipelining=0,
            print_params=False,
            print_search_space_size=False,
            quiet=False,
            results_log=None,
            results_log_details=None,
            seed_configuration=[],
            stop_after=None,
            technique=DEFAULT_TECHNIQUES,
            test_limit=5000,
        )

        # Setup some dummy classes required by opentuner to actually run.
        manipulator = self.build_manipulator(parameters_config)
        interface = DMI(args=args, manipulator=manipulator)
        self.api = TuningRunManager(interface, args)

    def build_manipulator(self, parameters_config):
        manipulator = ConfigurationManipulator()

        for pname in parameters_config:
            ptype = parameters_config[pname]["parameter_type"]
            assert ptype == 1, "Only double type is supported."
            pmin = parameters_config[pname]["double_min_value"]
            pmax = parameters_config[pname]["double_max_value"]
            param = FloatParameter(pname, pmin, pmax)
            manipulator.add_parameter(param)

        return manipulator

    def hashable_dict(self, d):
        """A custom function for hashing dictionaries.

        Parameters
        ----------
        d : dict or dict-like
            The dictionary to be converted to immutable/hashable type.

        Returns
        -------
        hashable_object : frozenset of tuple pairs
            Bijective equivalent to dict that can be hashed.
        """
        hashable_object = frozenset(d.items())
        return hashable_object

    def suggest(self, suggestion_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters.

        Args:
            suggestions_history: a list of historical suggestion parameters and rewards, in the form of
                    [[Parameter, Reward], [Parameter, Reward] ... ]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                        Reward: a float type value

                    The parameters and rewards of each iteration are placed in suggestions_history in the order of iteration.
                        len(suggestions_history) = n_suggestion * iteration(current number of iteration)

                    For example:
                        when iteration = 2, n_suggestion = 2, then
                        [[{'p1': 0, 'p2': 0, 'p3': 0}, -222.90621774147272],
                         [{'p1': 0, 'p2': 1, 'p3': 3}, -65.26678723205647],
                         [{'p1': 2, 'p2': 2, 'p3': 2}, 0.0],
                         [{'p1': 0, 'p2': 0, 'p3': 4}, -105.8151893979122]]

            n_suggestion: int, number of suggestions to return

        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}

                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        if suggestion_history:
            new_obs = suggestion_history[-n_suggestions:]
            for x_guess, y_ in new_obs:
                x_guess_ = self.hashable_dict(x_guess)

                # If we can't find the dr object then it must be the dummy guess.
                if x_guess_ not in self.x_to_dr:
                    assert x_guess == self.dummy_suggest, "Appears to be guess that did not originate from suggest"
                    continue

                # Get the corresponding DesiredResult object.
                dr = self.x_to_dr.pop(x_guess_, None)
                # This will also catch None from opentuner.
                assert isinstance(dr, DesiredResult), "DesiredResult object not available in x_to_dr"

                # Opentuner's arg names assume we are minimizing execution time.
                # So, if we want to maximize we have to pretend -y is a 'time'.
                result = Result(time=-y_)
                self.api.report_result(dr, result)

        # Update the n_suggestions if it is different from the current setting.
        if self.api.search_driver.args.parallelism != n_suggestions:
            self.api.search_driver.args.parallelism = n_suggestions
            warnings.warn("n_suggestions changed across suggest calls")

        # Require the user to already observe all previous suggestions.
        # Otherwise, opentuner will just recycle old suggestions.
        assert len(self.x_to_dr) == 0, "all the previous suggestions should have been observed by now"

        # The real meat of suggest from opentuner: Get next `n_suggestions`
        # unique suggestions.
        desired_results = [self.api.get_next_desired_result() for _ in range(n_suggestions)]

        # Save DesiredResult object in dict since observe will need it.
        X = []
        using_dummy_suggest = False
        for ii in range(n_suggestions):
            # Opentuner can give up, but the API requires guessing forever.
            if desired_results[ii] is None:
                assert self.dummy_suggest is not None, "opentuner gave up on the first call!"
                # Use the dummy suggestion in this case.
                X.append(self.dummy_suggest)
                using_dummy_suggest = True
                continue

            # Get the simple dict equivalent to suggestion.
            x_guess = desired_results[ii].configuration.data
            X.append(x_guess)

            # Now save the desired result for future use in observe.
            x_guess_ = self.hashable_dict(x_guess)
            assert x_guess_ not in self.x_to_dr, "the suggestions should not already be in the x_to_dr dict"
            self.x_to_dr[x_guess_] = desired_results[ii]
            # This will also catch None from opentuner.
            assert isinstance(self.x_to_dr[x_guess_], DesiredResult)

        assert len(X) == n_suggestions, "incorrect number of suggestions provided by opentuner"
        # Log suggestion for repeating if opentuner gives up next time. We can
        # only do this when it is not already being used since it we will be
        # checking guesses against dummy_suggest in observe.
        if not using_dummy_suggest:
            self.dummy_suggest = X[-1]

        next_suggestions = X

        return next_suggestions
