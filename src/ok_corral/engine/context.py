import json

class Context:

    SHARED_CONTEXT = "shared_context"
    ACTION_CONTEXT = "contexts"

    def __init__(self, p_context):

        if type(p_context) == str:
            p_context = json.loads(p_context)

        if type(p_context) == dict and self.ACTION_CONTEXT not in p_context:

            self.shared_context = p_context
            self.contexts = None
        # Contextes multiples

        else:

            #[self.SHARED_CONTEXT: contexte, self.ACTION_CONTEXT: [[action, contexte],...[action,contexte]] ]

            if type(p_context) == list:

                self.shared_context = None
                self.contexts = p_context

            else:

                self.shared_context = p_context[self.SHARED_CONTEXT] if self.SHARED_CONTEXT in p_context else None
                self.contexts = p_context[self.ACTION_CONTEXT]

    def get_context_by_action(self):

        return [[i_k, i_context if self.shared_context is None else {**self.shared_context, **i_context}] for i_k, i_context in self.contexts]
